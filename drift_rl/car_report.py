"""
car_report.py - policies x cars x tracks in server-style runs at one pace: speed, slide, rotation
(yaw acceleration, direction changes, sideways g), off course, walls, spinning, weaving. One JSON
line per run, then a summary per track and policy (trained cars against held-out cars).

    python car_report.py runs/gen8/policy.json,runs/gen9/policy.json spiritpeaksraceway,driftplayground_2021,magione \
        0.9 out.json --cars cars/car_pool.json [--tracks-dir DIR] [--skill 0.5]

Each track's <track>_pace_reference.csv in its folder is made by pace_model where missing. Each car
is asked for its own speed on each track (PACE_SCALE as setup gives it for a server:
drift_env.car_pace_scale, pace_model); --pace-scale 1 asks every car for the reference car's speed,
as reports before 2026-10-16 did.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, fast_sim as fs, drift_env as de, drive_report as dr, json
from policy import Policy
from dataclasses import replace
ap=argparse.ArgumentParser(); ap.add_argument('policies'); ap.add_argument('tracks'); ap.add_argument('pace', type=float)
ap.add_argument('out'); ap.add_argument('--cars', default='cars/car_pool.json'); ap.add_argument('--tracks-dir', default='tracks')
ap.add_argument('--skill', type=float, default=1.0); ap.add_argument('--pace-scale', type=float, default=None); ap.add_argument('--ticks', type=int, default=2400)
args=ap.parse_args()
policies=args.policies.split(','); tracks=args.tracks.split(','); pace=args.pace; out=args.out
pool=de.load_car_pool(args.cars)
rows=[]
for track in tracks:
    w=fs.load_world(args.tracks_dir,track=track)
    w.pace_ref=de.course_pace_reference(args.tracks_dir,track,w)
    for pf in policies:
        pol=Policy.load_json(pf)
        for name,car in pool:
            ps=args.pace_scale or de.car_pace_scale([w],car)
            car=dict(car,PACE_SCALE=ps)
            cfg,nr=de.server_run_configs(pol.meta,ticks=args.ticks,sliding_grip=0.55,pace=pace)
            cfg=replace(cfg,car_pool=(car,),recovery_start_fraction=0.0,skill_low=args.skill,skill_high=args.skill)
            env=de.DriftVecEnv(w,n_envs=12,env_config=cfg,reward_config=nr,seed=8)
            obs=env.reset(); b=env.base_params; 
            rec={k:[] for k in 'dt x z own latv yaw lat'.split()}; stats=np.zeros((12,de.EP_SIZE)); prev=env.states[:,fs.S_LATV].copy()
            br=car['DIST_CG_REAR_M']
            for _ in range(args.ticks):
                obs,_,term,trunc,_=env.step(pol(obs))
                s=env.states; dt=np.maximum(env.mids[:,de.M_DT],1e-3)
                rec['dt'].append(dt.copy())
                for k,i in (('x',fs.S_X),('z',fs.S_Z),('own',fs.S_OWN_SPEED),('latv',fs.S_LATV),('yaw',fs.S_YAW)): rec[k].append(s[:,i].copy())
                rec['lat'].append(((s[:,fs.S_LATV]-prev)/dt+s[:,fs.S_OWN_SPEED]*s[:,fs.S_YAW])/9.80665); prev=s[:,fs.S_LATV].copy()
                done=term|trunc; stats[done]=env.ep_out[done]
            r={k:np.array(v)[:-1] for k,v in rec.items()}
            slip=np.degrees(np.arctan2(r['latv']-br*r['yaw'],np.maximum(r['own'],0.5))); kmh=np.hypot(r['own'],r['latv'])*3.6
            rot=dr.pooled([dr.rotation(np.cumsum(r['dt'][:,i]),r['x'][:,i],r['z'][:,i],slip[:,i],kmh[:,i],r['lat'][:,i]) for i in range(12)])
            st=de.finalize_episode_stats(stats)
            row=dict(track=track,policy=pf.split('/')[-2],car=name,scale=round(ps,3),kmh=rot['kmh'],slide=rot['slide'],acc=rot['acc_p90'],rev=rot['reversals'],lat=rot['lat_p90'],
                     off=float(st['offcourse_fraction'].mean()),walls=float(st['pushout_tick_fraction'].mean()),spin=float(st['spin_recovery_fraction'].mean()),weave=float(st['weaves_per_min'].mean()))
            rows.append(row); print(json.dumps(row),flush=True)
json.dump(rows,open(out,'w'))
held=set()
for pf in policies:
    held|=set((Policy.load_json(pf).meta.get('cars') or {}).get('held_out') or [])
for tr in tracks:
    print(f"\n{tr}  pace {pace:g}, skill {args.skill:g}")
    print(f"{'':22s} {'km/h':>5s} {'slide':>6s} {'yawacc':>6s} {'chg/min':>7s} {'lat g':>5s} {'off':>6s} {'walls':>6s} {'spin':>6s} {'weave':>5s}  worst spinner")
    for pol in [p.split('/')[-2] for p in policies]:
        for grp in ('trained','held-out'):
            R=[r for r in rows if r['track']==tr and r['policy']==pol and ((r['car'] in held)==(grp=='held-out'))]
            if not R: continue
            m=lambda k: np.mean([r[k] for r in R]); worst=max(R,key=lambda r:r['spin'])
            print(f"{pol+' '+grp:22s} {m('kmh'):5.1f} {m('slide'):6.0%} {m('acc'):6.0f} {m('rev'):7.1f} {m('lat'):5.2f} {m('off'):6.1%} {m('walls'):6.1%} {m('spin'):6.1%} {m('weave'):5.1f}  {worst['car'][:24]} {worst['spin']:.1%}")
