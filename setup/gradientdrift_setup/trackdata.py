"""A check that the track tools setup carries work where setup runs (the packaged program's CI):
a generated course is made and run through the same steps setup_track.py runs on a real track's
data, compiled code (numba) included. `gradient-drift-setup selftest`; not listed in the help."""
import os
import tempfile
import time


def selftest():
    from . import game  # noqa: F401  (puts drift_rl on the path)
    import generate_track as gt
    import line_tools as lt
    import fast_sim as fs
    import setup_track as st
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tracks:
        name = gt.ensure(tracks, 1, 0)
        print(f'generated {name} ({time.time() - t0:.0f} s)')
        for module in ('build_road_mesh', 'build_course', 'build_zones', 'sensor_view'):
            st.run(module, ['--tracks', tracks, '--track', name])
        world = fs.load_world(tracks, track=name, fine_margin_m=40.0, fine_cell_size=4.0, rm_cell=6.0)
        fs.require_road_mesh(world, tracks, name)
        fs.require_course(world, tracks, name)
        pos, _ = lt.read_line(os.path.join(tracks, name, f'{name}_fast_lane.ai'))
        rep = lt.check(pos)
        cen = lt.centring(world, pos)
        print(f"line: {rep['points']} points, {rep['length_m']:.0f} m; off the middle {cen['off_centre_median_m']:.2f} m")
        import check_track
        problems = check_track.check(tracks, name)
        print('check: ' + ('; '.join(problems) if problems else 'speed targets and walls look right'))
    print(f'track tools OK ({time.time() - t0:.0f} s)')
    return 0
