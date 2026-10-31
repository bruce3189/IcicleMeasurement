"""Shared window evaluation: first-pass measurement or full diagnostics."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .models import DEPTH_MIN_M, DEPTH_MAX_M
from .core import otsu_split, detection_problem
from .near_cluster import separation


@dataclass
class WindowAnalysisConfig:
    radii: tuple = (4, 5, 8, 12, 18, 25, 35, 45, 60)
    near_min_exclusive: int = 26
    far_min_exclusive: int = 46
    separation_min_exclusive: float = 3.0
    mad_multiplier: float = 3.5
    mad_floor_m: float = 0.001

    def validate(self):
        if (not self.radii or any(type(r) is not int or r<0 for r in self.radii)
                or list(self.radii)!=sorted(set(self.radii))):
            raise ValueError('半径序列须为递增非负整数（r=0对应1×1）')
        for name in ('near_min_exclusive','far_min_exclusive'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:
                raise ValueError(name+'须为非负整数')
        for name in ('separation_min_exclusive','mad_multiplier','mad_floor_m'):
            value=getattr(self,name)
            if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<=0:
                raise ValueError(name+'须为有限正数')


def candidate_checks(near_n, far_n, S, config):
    return {'near_samples':near_n>config.near_min_exclusive,
            'far_samples':far_n>config.far_min_exclusive,
            'separation':S is not None and S>config.separation_min_exclusive}


def windows_to_short_side(shape, bbox, point, radii):
    """Enumerate unique image-clipped rectangles through the geometric cap."""
    height,width=shape
    x1,y1,x2,y2=bbox;u,v=point
    cap=max(0,math.floor((max(1.,min(x2-x1,y2-y1))-1)/2))
    # Include the cap even if it lies beyond the last configured radius.
    actual=sorted({min(r,cap) for r in radii}|{cap})
    rectangles={}
    for r in actual:
        bounds=(max(0,math.floor(u-r)),max(0,math.floor(v-r)),
                min(width,math.floor(u+r)+1),min(height,math.floor(v+r)+1))
        if bounds not in rectangles:rectangles[bounds]=r
    return [(r,bounds) for bounds,r in rectangles.items()]


def group_statistics(near,far):
    stats={}
    for name,values in [('near',near),('far',far)]:
        stats[name+'_n']=int(len(values))
        stats[name+'_mean_m']=float(values.mean()) if len(values) else None
        stats[name+'_std_m']=float(values.std()) if len(values) else None
    stats['S']=separation(near,far) if len(near) and len(far) else None
    return stats


def analyze_endpoint(frame,detection,index,config,cancel=lambda:False,*,stop_on_first_candidate=False,clusterer=None):
    clusterer=clusterer or otsu_split
    point=detection.keypoints[index];height,width=frame.depth_m.shape
    endpoint={'name':('tip','base')[index],'pixel':list(point),'windows':[],
              'status':'diagnostic_only','score':None,'selected_window':None,
              'depth_m':None,'xyz_m':None,'candidate_count':0}
    locked=set();previous_mean=None
    for radius,bounds in windows_to_short_side(frame.depth_m.shape,detection.bbox,point,config.radii):
        if cancel():endpoint['status']='cancelled';return endpoint
        left,top,right,bottom=bounds
        yy,xx=np.mgrid[top:bottom,left:right]
        values=frame.depth_m[top:bottom,left:right].astype(float).ravel()
        valid=np.isfinite(values)&(values>=DEPTH_MIN_M)&(values<=DEPTH_MAX_M)&(values>0)
        ids=(yy*width+xx).ravel()[valid];z=values[valid]
        labels=clusterer(z,config);near,far=labels==1,labels==2
        record={'index':len(endpoint['windows']),'radius_px':radius,'bounds':list(bounds),
                'width_px':right-left,'height_px':bottom-top,'area_n':int(len(values)),
                'valid_n':int(len(z)),'invalid_n':int((~valid).sum()),
                'valid_ids':ids.tolist(),
                'mad_removed_n':int((labels==0).sum()),
                'mad_removed_initial_n':int((labels==0).sum()),
                'mad_removed_internal_n':0,
                'unselected_n':int((labels!=0).sum()),
                'basic':group_statistics(z[near],z[far]),'selected':None,
                'internal_split':False,'identity_overlap_n':None,'identity_locked_before':bool(locked),
                'target_ids':[],'reference_ids':[],'mean_change_m':None,
                'checks':candidate_checks(0,0,None,config),'candidate':False,
                'score':None,'score_status':'不评分','reasons':[]}
        endpoint['windows'].append(record)
        if not near.any() or not far.any():
            record['reasons']=['未形成可比较的两个深度簇'];continue
        candidates=[near,far]
        if locked:
            overlap=[len(locked.intersection(ids[m].tolist())) for m in candidates]
            record['identity_overlap_n']=max(overlap)
            if max(overlap)==0:
                record['reasons']=['无法延续原簇'];continue
            target=candidates[int(np.argmax(overlap))]
        else:target=candidates[0]
        mean=float(z[target].mean())
        farther=[m for m in candidates if m is not target and m.any() and z[m].mean()>mean]
        reference=np.logical_or.reduce(farther) if farther else np.zeros(len(z),bool)
        selected=group_statistics(z[target],z[reference]);record['selected']=selected
        record['unselected_n']=len(z)-record['mad_removed_n']-selected['near_n']-selected['far_n']
        record['target_ids']=ids[target].tolist();record['reference_ids']=ids[reference].tolist()
        record['mean_change_m']=abs(mean-previous_mean) if previous_mean is not None else None
        record['checks']=candidate_checks(selected['near_n'],selected['far_n'],selected['S'],config)
        record['candidate']=all(record['checks'].values())
        names={'near_samples':'近簇须>%d'%config.near_min_exclusive,'far_samples':'远簇须>%d'%config.far_min_exclusive,
               'separation':'S须>%g'%config.separation_min_exclusive}
        record['reasons']=[names[k] for k,v in record['checks'].items() if not v]
        if selected['near_n']>config.near_min_exclusive and not locked:locked=set(record['target_ids'])
        previous_mean=mean
        if stop_on_first_candidate and record['candidate']:
            endpoint['candidate_count']=1
            endpoint['status']='first_candidate_found'
            return endpoint
    endpoint['candidate_count']=sum(r['candidate'] for r in endpoint['windows'])
    if not endpoint['candidate_count']:endpoint['status']='no_candidate'
    return endpoint


def analyze_frame_windows(frame,detections,config=None,cancel=None):
    """Return diagnostics only. Never select a window or calculate a length."""
    config=config or WindowAnalysisConfig();config.validate();frame.validate()
    cancel=cancel or (lambda:False)
    result={'kind':'all_window_analysis','version':1,'status':'diagnostic_only',
            'config':asdict(config),'effective_valid_range_m':[DEPTH_MIN_M,DEPTH_MAX_M],
            'source_valid_range_m':[frame.min_depth_m,frame.max_depth_m],
            'scoring_rule':None,'selected_window':None,'distance_m':None,'objects':[]}
    for detection in detections:
        if cancel():result['status']='cancelled';break
        problem=detection_problem(detection,frame.depth_m.shape)
        obj={'id':detection.id,'bbox':list(detection.bbox),'status':'diagnostic_only',
             'distance_m':None,'endpoints':[]}
        if problem:obj.update(status='skipped',reason=problem)
        else:
            for index in (0,1):
                ep=analyze_endpoint(frame,detection,index,config,cancel)
                obj['endpoints'].append(ep)
                if ep['status']=='cancelled':obj['status']='cancelled';result['status']='cancelled';break
            if obj['status']!='cancelled' and any(e['status']=='no_candidate' for e in obj['endpoints']):obj['status']='no_candidate'
        result['objects'].append(obj)
        if result['status']=='cancelled':break
    return result
