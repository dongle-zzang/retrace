from datetime import datetime, timedelta, timezone
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import select

from backend.app.db import ParkingEvent, ParkingSpace
from backend.app.occupancy import OccupancyEvaluator, ParkingPolicy, area, contains, intersection_area
from backend.app.schemas import FrameIn
from backend.app.ingest import utcnow
from test_backend import service, frame, SESSION
from test_parking import create, BASE

AT = datetime(2026, 1, 1, tzinfo=timezone.utc)
POLYGON = [{"x": .1, "y": .55}, {"x": .5, "y": .55}, {"x": .5, "y": .8}, {"x": .1, "y": .8}]
VALIDATED = ParkingPolicy(validated_cameras=("first",))


def vehicle(x=.15, track=9, confidence=.9, tracker_confidence=.9):
    return {"track_id": track, "class_id": 0, "confidence": confidence,
            "tracker_confidence": tracker_confidence,
            "bbox": {"x": x * 1920, "y": .2 * 1080, "width": .3 * 1920, "height": .55 * 1080}}


def observation(t, vehicles=None, **values):
    return FrameIn.model_validate({**frame(at=AT + timedelta(seconds=t)), "frame_number": int(t * 10) + 1,
        "vehicles": [vehicle()] if vehicles is None else vehicles, "vehicle_detection_enabled": True,
        "vehicle_inference_done": True, **values})


def slot(key="s", polygon=None, revision=1):
    return NS(space_id=key, camera_id="first", polygon=polygon or POLYGON, revision=revision)


def state(evaluator, t, vehicles=None, spaces=None, **values):
    return evaluator.observe(observation(t, vehicles, **values), SESSION, spaces or [slot()])[0]


def test_geometry_concave_clip_and_anchor():
    polygon = [(0, 0), (1, 0), (1, .5), (.5, .5), (.5, 1), (0, 1)]
    assert area(polygon) == .75
    assert intersection_area(polygon, (0, 0, 1, 1)) == .75
    assert intersection_area(polygon, (.25, .25, .75, .75)) == pytest.approx(.1875)
    assert intersection_area(polygon, (2, 2, 3, 3)) == 0
    assert contains(polygon, (0, 0)) and not contains(polygon, (.75, .75))


def occupy(evaluator, until=10):
    for t in range(0, until + 1, 2):
        result = state(evaluator, t)
    assert result.occupancy == "occupied"


def test_stationary_vehicle_dwell_and_grace_recovery():
    evaluator = OccupancyEvaluator(VALIDATED)
    for t in range(0, 10, 2):
        assert state(evaluator, t).occupancy == "unknown"
    assert state(evaluator, 10).occupancy == "occupied"
    assert state(evaluator, 12, []).reason == "vacancy_grace"
    assert state(evaluator, 14).occupancy == "occupied"
    for t in range(16, 28, 2):
        assert state(evaluator, t, []).occupancy == "occupied"
    # A vanished occupant that was never seen leaving is unknown, not empty.
    result = state(evaluator, 28, [])
    assert (result.occupancy, result.reason) == ("unknown", "occupant_lost")
    assert result.evidence["occupant_lost"] and result.evidence["camera_validated"]
    for t in range(30, 76, 2):
        assert state(evaluator, t, []).occupancy == "unknown"
    result = state(evaluator, 76, [])
    assert (result.occupancy, result.reason) == ("empty", "prolonged_absence")


def test_parked_vehicle_missed_briefly_stays_occupied_and_recovers():
    evaluator = OccupancyEvaluator(VALIDATED)
    occupy(evaluator)
    for t, present in ((12, False), (14, False), (16, True), (18, False), (20, False), (22, False), (24, True)):
        assert state(evaluator, t, None if present else []).occupancy == "occupied"
    # A longer miss turns unknown. The same unmoved track restores occupancy at once;
    # a new track ID (NvDCF usually re-identifies after a long miss) must dwell again.
    for t in range(26, 40, 2):
        result = state(evaluator, t, [])
    assert (result.occupancy, result.reason) == ("unknown", "occupant_lost")
    assert state(evaluator, 40).occupancy == "occupied"
    for t in range(42, 56, 2):
        state(evaluator, t, [])
    for t in range(56, 66, 2):
        assert state(evaluator, t, [vehicle(track=11)]).occupancy == "unknown"
    assert state(evaluator, 66, [vehicle(track=11)]).occupancy == "occupied"


def test_observed_departure_becomes_empty_after_short_absence():
    for leaving in ([vehicle(x=.2)], [vehicle(x=.75)]):  # occupant moves, or is tracked outside
        evaluator = OccupancyEvaluator(VALIDATED)
        occupy(evaluator)
        result = state(evaluator, 12, leaving)
        assert result.occupancy != "empty" and result.evidence["departure_observed"]
        assert state(evaluator, 14, []).occupancy != "empty"
        assert state(evaluator, 16, []).occupancy != "empty"
        result = state(evaluator, 20, [])
        assert (result.occupancy, result.reason) == ("empty", "departure_observed")


def test_absence_on_unvalidated_camera_never_confirms_empty():
    evaluator = OccupancyEvaluator()
    for t in range(0, 120, 2):
        result = state(evaluator, t, [])
        assert (result.occupancy, result.reason) == ("unknown", "absence_unverified_camera")
    evaluator = OccupancyEvaluator(VALIDATED)
    assert [state(evaluator, t, []).occupancy for t in range(0, 8, 2)] == ["unknown"] * 3 + ["empty"]


def test_low_confidence_candidate_in_space_blocks_empty():
    evaluator = OccupancyEvaluator(VALIDATED)
    for t in range(0, 60, 2):
        result = state(evaluator, t, [vehicle(confidence=.3)])
        assert (result.occupancy, result.reason) == ("unknown", "ambiguous_or_low_confidence")


def test_track_id_switch_on_parked_vehicle_then_miss_is_not_quick_empty():
    evaluator = OccupancyEvaluator(VALIDATED)
    occupy(evaluator)
    assert state(evaluator, 12, [vehicle(track=10)]).occupancy == "unknown"
    for t in range(14, 30, 2):
        result = state(evaluator, t, [])
        assert result.occupancy == "unknown" and result.evidence["occupant_lost"]


def test_passing_slow_creeping_and_track_switch_do_not_confirm():
    for speed in (.025, .008):
        evaluator = OccupancyEvaluator(ParkingPolicy(stationary_seconds=6, max_displacement=.02))
        for t in range(0, 8, 2):
            assert state(evaluator, t, [vehicle(x=.15 + speed * t)]).occupancy != "occupied"
    evaluator = OccupancyEvaluator()
    for t in range(0, 14, 2):
        assert state(evaluator, t, [vehicle(track=t)]).occupancy != "occupied"


def test_bbox_overlap_above_space_does_not_prove_occupied():
    evaluator = OccupancyEvaluator()
    wrong = vehicle()
    wrong["bbox"]["height"] = .3 * 1080  # Entire bbox touches spot, lower footprint ends outside.
    for t in range(0, 14, 2):
        assert state(evaluator, t, [wrong]).occupancy != "occupied"


def test_duplicate_outoforder_gap_generation_and_polygon_reset():
    evaluator = OccupancyEvaluator()
    first = observation(0)
    evaluator.observe(first, SESSION, [slot()])
    assert evaluator.observe(first, SESSION, [slot()]) == []
    assert evaluator.observe(observation(-1, frame_number=0), SESSION, [slot()]) == []
    assert state(evaluator, 11).occupancy == "unknown"  # Missing observations do not count as dwell.
    for t in range(13, 22, 2):
        result = state(evaluator, t)
    assert result.occupancy == "occupied"
    assert state(evaluator, 23, generation=2).occupancy == "unknown"
    assert state(evaluator, 25, spaces=[slot(revision=2)]).occupancy == "unknown"


@pytest.mark.parametrize("values", [{"track": None}, {"confidence": None}, {"confidence": .1},
                                   {"tracker_confidence": None}, {"tracker_confidence": .1}])
def test_uncertain_vehicles_cannot_prove_vacancy(values):
    evaluator = OccupancyEvaluator()
    for t in range(0, 16, 2):
        assert state(evaluator, t, [vehicle(**values)]).occupancy == "unknown"


def test_one_vehicle_is_not_assigned_to_two_ambiguous_spaces():
    evaluator = OccupancyEvaluator()
    spaces = [slot("a"), slot("b")]
    for t in range(0, 14, 2):
        results = evaluator.observe(observation(t), SESSION, spaces)
        assert [result.occupancy for result in results] == ["unknown", "unknown"]


def test_disabled_inference_immediately_unknown_and_empty_requires_observations():
    evaluator = OccupancyEvaluator(VALIDATED)
    occupy(evaluator)
    result = state(evaluator, 12, vehicle_inference_done=False)
    assert (result.occupancy, result.reason) == ("unknown", "detector_unavailable")
    assert state(evaluator, 14, vehicle_detection_enabled=False).occupancy == "unknown"
    # The car may still be parked: absence after an outage needs the long rule.
    for t in range(16, 76, 2):
        assert state(evaluator, t, []).occupancy == "unknown"
    assert state(evaluator, 76, []).occupancy == "empty"


@pytest.mark.parametrize("values", [{"footprint_height": 0}, {"horizontal_inset": .5},
    {"exit_overlap": .9}, {"min_confidence": 2}, {"max_speed": float("nan")},
    {"max_observation_gap": 0}, {"require_anchor": "true"}, {"validated_cameras": "first"},
    {"validated_cameras": [""]}, {"min_empty_observations": 0}, {"min_empty_observations": 1.5},
    {"lost_empty_seconds": 1}])
def test_policy_validation(values):
    with pytest.raises(ValueError):
        ParkingPolicy(**values)


def test_policy_environment_empty_default_and_override(monkeypatch):
    from backend.app.settings import Settings
    monkeypatch.setenv("POSTGRES_PASSWORD", "synthetic-password")
    monkeypatch.setenv("PARKING_POLICY", "")
    assert Settings.from_env().parking_policy.stationary_seconds == 10
    monkeypatch.setenv("PARKING_POLICY", '{"stationary_seconds":15,"require_anchor":false}')
    policy = Settings.from_env().parking_policy
    assert policy.stationary_seconds == 15 and policy.require_anchor is False
    monkeypatch.setenv("PARKING_POLICY", '{"validated_cameras":["first"]}')
    assert Settings.from_env().parking_policy.validated_cameras == ("first",)


def test_polling_db_events_live_vehicles_and_unknown_failure(service, monkeypatch):
    client, app, upstream, _, _ = service
    space = create(client, polygon=POLYGON).json()
    started = utcnow() - timedelta(seconds=1)
    clock = [started]
    monkeypatch.setattr("backend.app.ingest.utcnow", lambda: clock[0])
    monkeypatch.setattr("backend.app.main.utcnow", lambda: clock[0])
    for t in range(0, 12, 2):
        clock[0] = started + timedelta(seconds=t)
        item = observation(t).model_dump(mode="json")
        item["timestamp"] = clock[0].isoformat()
        upstream["metadata"]["frames"] = [item]
        app.state.poller.poll_once()
    assert client.get(f"{BASE}/{space['space_id']}").json()["occupancy"] == "occupied"
    assert client.get("/api/cameras/first/vehicles").json()["vehicles"][0]["track_id"] == "9"
    assert client.get("/api/cameras/first/tracks").json()["current"][0]["track_id"] == "42"
    snapshot = client.get("/api/parking").json()
    assert snapshot["cameras"][0]["summary"]["occupied"] == 1
    events = client.get("/api/parking/events").json()
    assert len(events["events"]) == 1 and events["events"][0]["occupancy"] == "occupied"
    assert events["events"][0]["evidence"]["generation"] == 1
    app.state.poller.poll_once()  # duplicate doesn't create a transition or extend observation time
    assert len(client.get("/api/parking/events").json()["events"]) == 1
    assert client.get("/api/cameras/first/parking-assessments").json()["assessments"][0]["reason"] == "stationary_vehicle"
    clock[0] += timedelta(seconds=1)
    upstream["error"] = True
    app.state.poller.poll_once()
    assert client.get(f"{BASE}/{space['space_id']}").json()["occupancy"] == "unknown"
    remaining = client.get(f"/api/parking/events?after={events['cursor']}").json()
    assert remaining["events"][0]["previous_occupancy"] == "occupied"
    assert remaining["events"][0]["reason"] == "camera_or_metadata_unavailable"
    assert client.get("/api/cameras/first/vehicles").json()["vehicles"] == []


def test_vehicles_api_passes_outline_and_one_bad_outline_never_drops_the_frame(service):
    client, app, upstream, _, _ = service
    square = [[100.0, 200.0], [400.0, 200.0], [400.0, 500.0], [100.0, 500.0]]
    bad = [[[1, 2]] * 33, [[1, 2], [3, 4]], "bad", [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2], [3, "x"], [5, 6]],
           [[1, 2], [True, False], [5, 6]], None]
    vehicles = [vehicle(track=1) | {"outline": square}, vehicle(track=2)]
    vehicles += [vehicle(track=3 + i) | {"outline": value} for i, value in enumerate(bad)]
    upstream["metadata"]["frames"] = [{**frame(), "persons": [], "vehicles": vehicles,
                                       "vehicle_detection_enabled": True, "vehicle_inference_done": True}]
    app.state.poller.poll_once()
    body = client.get("/api/cameras/first/vehicles").json()["vehicles"]
    assert [v["track_id"] for v in body] == [str(i) for i in range(1, 3 + len(bad))]
    assert [v["outline"] for v in body] == [square] + [None] * (1 + len(bad))
    assert body[0]["bbox"] == vehicle()["bbox"]


def test_vehicle_outline_with_non_finite_points_is_null():
    from backend.app.schemas import VehicleIn
    for value in (float("nan"), float("inf")):
        assert VehicleIn.model_validate(vehicle() | {"outline": [[1, 2], [3, value], [5, 6]]}).outline is None
    assert VehicleIn.model_validate(vehicle() | {"outline": ((1, 2), (3, 4), (5, 6.5))}).outline == [
        (1.0, 2.0), (3.0, 4.0), (5.0, 6.5)]


def test_current_api_expires_without_poller_and_geometry_edit_event(service, monkeypatch):
    client, app, upstream, _, _ = service
    space = create(client, polygon=POLYGON).json()
    now = utcnow()
    from backend.app.parking import update_occupancy
    with app.state.sessions.begin() as session:
        update_occupancy(session, "first", space["space_id"], 1, "occupied", now)
    monkeypatch.setattr("backend.app.main.utcnow", lambda: now + timedelta(seconds=11))
    assert client.get(f"{BASE}/{space['space_id']}").json()["occupancy"] == "unknown"
    assert client.get("/api/cameras/first/parking-summary").json()["unknown"] == 1
    changed = [dict(p) for p in POLYGON]
    changed[0]["x"] = .05
    assert client.patch(f"{BASE}/{space['space_id']}", json={"polygon": changed}).status_code == 200
    assert client.get("/api/parking/events").json()["events"][-1]["reason"] == "polygon_changed"


def test_failed_transaction_does_not_publish_evaluator_memory(service, monkeypatch):
    _, app, upstream, _, _ = service
    create(service[0], polygon=POLYGON)
    item = observation(0).model_dump(mode="json")
    item["timestamp"] = utcnow().isoformat()
    upstream["metadata"]["frames"] = [item]
    original = app.state.poller.occupancy
    def fail(*args, **kwargs):
        raise RuntimeError("Synthetic transaction failure")
    monkeypatch.setattr("backend.app.ingest.set_decision", fail)
    with pytest.raises(RuntimeError):
        app.state.poller.poll_once()
    assert app.state.poller.occupancy is original and not original.memory
    with app.state.sessions() as session:
        assert len(list(session.scalars(select(ParkingEvent)))) == 0


def test_labelled_replay_reports_aggregate_metrics(tmp_path):
    import json
    from tools.replay_parking import replay
    records = []
    for t in range(0, 12, 2):
        records.append({"frame": observation(t).model_dump(mode="json"), "runtime_session": SESSION,
                        "spaces": [{"space_id": "s", "camera_id": "first", "revision": 1, "polygon": POLYGON}],
                        "expected": {"s": "occupied" if t == 10 else "unknown"},
                        "vehicle_labels": [{"x": .15, "y": .2, "width": .3, "height": .55}]})
    path = tmp_path / "synthetic.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in records))
    result = replay(path)
    assert result["frames"] == 6 and result["occupancy_labels"] == 6
    assert result["occupancy_accuracy"] == 1
    assert result["vehicle_tp"] == 6 and result["vehicle_fp"] == result["vehicle_fn"] == 0
