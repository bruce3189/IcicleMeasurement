"""python -m icicle_measure.benchmark --output outputs/rgbd_validation"""
import argparse
from pathlib import Path
from .core import measure_frame
from .models import Config
from .simulator import generate, SCENARIOS
from .io import write_json, save_simulation, export_results


def benchmark(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    trials, z_errors, length_errors = [], [], []
    for seed in range(30):
        s = generate('noisy', seed)
        obj = measure_frame(s.frame, s.detections, s.intrinsics).objects[0]
        row = {'seed': seed, 'status': obj.status}
        if obj.distance_m is not None:
            errors = [abs(ep.depth_m - s.truth[obj.id]['depth_m'][i]) * 1000 for i, ep in enumerate(obj.endpoints)]
            length_error = abs(obj.distance_m - s.truth[obj.id]['distance_m']) * 1000
            z_errors.extend(errors)
            length_errors.append(length_error)
            row.update(endpoint_errors_mm=errors, length_error_mm=length_error)
        trials.append(row)
    scenes = {}
    for name in SCENARIOS:
        s = generate(name)
        result = measure_frame(s.frame, s.detections, s.intrinsics)
        scenes[name] = [{'id': obj.id, 'status': obj.status, 'reason': obj.reason} for obj in result.objects]
    report = {'noise_sigma_mm': 3, 'seeds': list(range(30)), 'accepted': len(length_errors), 'total': 30,
              'acceptance_rate': len(length_errors) / 30,
              'max_endpoint_error_mm': max(z_errors) if z_errors else None,
              'max_length_error_mm': max(length_errors) if length_errors else None,
              'trials': trials, 'scenarios': scenes,
              'limitations': 'Synthetic locally flat endpoint surfaces; does not establish transparent-ice sensor accuracy.'}
    rejection_cases = ('sparse', 'overlap', 'background', 'empty', 'drift', 'membership', 'interference',
                       'touching_surfaces')
    slope = generate('slope_bias')
    approximate = measure_frame(slope.frame, slope.detections, slope.intrinsics).objects[0]
    strict = measure_frame(slope.frame, slope.detections, slope.intrinsics,
                           Config(endpoint_depth_mode='strict_local')).objects[0]
    report['slope_mode_comparison'] = {
        'near_cluster_mean_status': approximate.status, 'strict_local_status': strict.status,
        'approximate_endpoint_error_mm': [abs(ep.depth_m - slope.truth[approximate.id]['depth_m'][i]) * 1000
                                          if ep.depth_m is not None else None
                                          for i, ep in enumerate(approximate.endpoints)],
        'note': 'Known surface gradient; near-mean approximation retains its bias and advisory. No YOLO was run.'}
    report['rejection_checks_passed'] = all(
        all(obj['status'] == 'rejected' for obj in scenes[name]) for name in rejection_cases)
    report['passed'] = bool(len(length_errors) >= 27 and max(z_errors) <= 10 and max(length_errors) <= 20
                            and report['rejection_checks_passed'])
    write_json(root / 'benchmark.json', report)
    s = generate('multi')
    save_simulation(root / 'example_input', s)
    result = measure_frame(s.frame, s.detections, s.intrinsics)
    export_results(root / 'example_result', s.frame, s.detections, result)
    print('Accepted: %d/30; max endpoint error: %.4f mm; max length error: %.4f mm' % (
        len(length_errors), max(z_errors) if z_errors else float('nan'), max(length_errors) if length_errors else float('nan')))
    print('Report:', root / 'benchmark.json')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/rgbd_validation')
    args = parser.parse_args()
    report = benchmark(args.output)
    raise SystemExit(0 if report['passed'] else 1)
