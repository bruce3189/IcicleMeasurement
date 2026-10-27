"""Two explicit, image-reviewed scene hypotheses; not automatic wall recognition."""
from dataclasses import replace
import time
import copy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .pseudo_depth import (ROOT, load_project, scene_from_inference, compose_scene,
                           export_project, fixed_depth_rgb, depth_colorbar)
from .pseudo_models import run_inference
from .pseudo_gui import scene_preview
from .core import measure_frame
from .io import write_json, export_results


def main():
    output = ROOT / 'outputs' / 'pseudo_depth_refined'
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    rows = []
    planes = {
        '002_00002': [
            ([[0,87],[775,87],[750,634],[0,634]], [1.8,2.1,1.85]),
            ([[775,87],[880,87],[860,634],[750,634]], [2.1,2.25,2.1]),
            ([[880,87],[1279,87],[1279,634],[860,634]], [2.25,1.9,2.25])],
        '002_00041': [
            ([[0,87],[920,87],[870,634],[0,634]], [1.8,2.1,1.85]),
            ([[920,87],[1050,87],[1010,634],[870,634]], [2.1,2.25,2.1]),
            ([[1050,87],[1279,87],[1279,634],[1010,634]], [2.25,1.95,2.25])]
    }
    for stem, definitions in planes.items():
        start = time.perf_counter()
        old = load_project(ROOT / 'outputs' / 'pseudo_depth_samples' / stem)
        cfg = replace(old.config, generation_version=2)
        coords = ({'7':[(345,300,True),(314,300,False)], '8':[(314,300,True),(345,300,False)]}
                  if stem == '002_00002' else
                  {'6':[(327,260,True),(375,260,False)], '7':[(398,260,True),(369,260,False)],
                   '8':[(363,300,True),(393,300,False),(327,260,False)]})
        prompts = {key:[{'x':x,'y':y,'positive':positive} for x,y,positive in values] for key,values in coords.items()}
        path = run_inference(old.scene.source_rgb, old.scene.source_labels, cfg, prompts=prompts,
                             progress=lambda message: print(message.get('message'), flush=True))
        scene = scene_from_inference(old.scene.source_rgb, old.scene.source_labels, path, cfg)
        scene.edits['prompts'] = prompts
        automatic = compose_scene(scene, cfg)
        if stem == '002_00041':
            for i,d in enumerate(scene.detections):
                if d.id in ('9','10','11'):
                    scene.masks[i][:] = False
                    scene.edits['actions'].append({'kind':'manual_exclude_detection','id':d.id,
                        'reason':'RGB局部检查：左侧三根保留6/7/8；额外重复候选不生成表面，原记录保留'})
        # Explicitly authored planar-wall regions, selected by inspecting the RGB.
        # Anchor depths and common ice baseline are synthetic choices, not recovered truth.
        scene.edits['background_planes'] = [dict(polygon=p, depths_m=z, source='manual_RGB_review_hypothesis') for p,z in definitions]
        scene.edits['background_planes'][0]['protected_polygons'] = (
            [[[0,87],[24,87],[24,634],[0,634]],
             [[250,87],[265,87],[269,122],[286,133],[742,112],[759,87],[779,87],[773,125],[750,139],[284,150],[255,137]],
             [[137,479],[226,461],[232,530],[140,538]],
             [[0,556],[39,551],[69,579],[51,607],[0,606]]]
            if stem == '002_00002' else
            [[[123,87],[154,87],[83,634],[51,634]],
             [[0,532],[78,537],[119,525],[143,550],[135,578],[87,590],[0,569]],
             [[226,481],[316,470],[321,525],[226,535]]])
        scene.edits['objects'] = {d.id: {'base_m':1.25, 'slope_m':0.} for d in scene.detections}
        scene.edits['actions'].append({'kind':'scene_review', 'description':
            '人工检查 RGB 圈选墙面；三个墙面分区。深度锚值是假设，共用冰凌基线1.25m仅作对照；不按长度拟合；没有自动识别墙面。'})
        result = compose_scene(scene, cfg)
        target = output / stem
        export_project(target, result)
        statuses = {}
        for name, frame in [('complete',result.clean_frame),('observed',result.frame)]:
            measured = measure_frame(frame, scene.detections, scene.intrinsics)
            export_results(target / ('measurement_' + name), frame, scene.detections, measured)
            statuses[name] = [{'id':o.id,'status':o.status,'reason':o.reason,'distance_m':o.distance_m} for o in measured.objects]
        write_json(target / 'timing_and_measurement.json', {'measurement':statuses,'seconds':time.perf_counter()-start})
        np.save(target / 'automatic_v2_before_manual_m.npy', automatic.clean_frame.depth_m)
        images = [('原 RGB / 原预测',scene_preview(result,'rgb')),
                  ('新版掩膜 / 紫色重叠仍待修',scene_preview(result,'masks')),
                  ('旧版完整表面',fixed_depth_rgb(old.clean_frame.depth_m)),
                  ('新版表面 + 人工分割提示（背景未修）',fixed_depth_rgb(automatic.clean_frame.depth_m)),
                  ('人工墙面约束 + 假想冰凌表面',fixed_depth_rgb(result.clean_frame.depth_m)),
                  ('独立观测缺陷：空洞 / 背景回波',fixed_depth_rgb(result.frame.depth_m))]
        panel = Image.new('RGB',(1280,1240),'#101c2b'); draw=ImageDraw.Draw(panel)
        draw.text((12,10),stem+' · 模型伪深度 / 人工场景假设，非真实深度',font=font,fill='#ffe2a1')
        for i,(title,array) in enumerate(images):
            x,y=(i%2)*640,45+(i//2)*385
            draw.text((x+6,y),title,font=font,fill='#dce6ef')
            panel.paste(Image.fromarray(array).resize((630,354)),(x+5,y+26))
        panel.paste(Image.fromarray(depth_colorbar()),(410,1196))
        panel.save(target/'comparison.jpg',quality=95)
        # Equal-aspect camera-coordinate cloud; no Z exaggeration or class-specific scaling.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(12,5))
        for i,(label,depth) in enumerate([('Previous',old.clean_frame.depth_m),('Scene hypothesis',result.clean_frame.depth_m)]):
            yy,xx=np.indices(depth.shape); take=(depth>0)&(xx%6==0)&(yy%6==0)
            z=depth[take]; k=scene.intrinsics
            x=(xx[take]-k.cx)*z/k.fx; y=(yy[take]-k.cy)*z/k.fy
            ax=fig.add_subplot(1,2,i+1,projection='3d')
            ax.scatter(x,z,-y,c=z,s=1,cmap='turbo',vmin=.5,vmax=4)
            ax.set(xlabel='camera X (m)',ylabel='camera Z (m)',zlabel='-camera Y (m)',title=label)
            limits=((-2.5,2.5),(.3,4.1),(-1.5,1.5))
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
            ax.set_box_aspect((5,3.8,3)); ax.view_init(elev=15,azim=-65)
        fig.suptitle('SYNTHETIC ONLY / same metric aspect and depth colour range'); fig.tight_layout()
        fig.savefig(target/'cloud_comparison.png',dpi=150); plt.close(fig)
        rows.append({'stem':stem,'seconds':time.perf_counter()-start,'overlap_pixels':result.report['overlap_pixels'],
                     'mask_review_ids':[d.id for d,m in zip(scene.detections,scene.model_info['segmentation']) if m.get('needs_review')],
                     'manual_background_planes':len(definitions),'all_ice_baselines_manually_assumed_m':1.25})
        print(stem,rows[-1],flush=True)
    write_json(output/'review_report.json',{'samples':rows,'truth_available':False,'bulk_400_run':False})
    html='<meta charset="utf-8"><title>改进深度模拟</title><style>body{background:#101c2b;color:#eef;font:18px system-ui;margin:24px}img{max-width:100%}a{color:#9df}</style>'
    html+='<h1>两张样例：改前 / 自动改进 / 人工场景约束</h1><p>非实测。墙面区域经原图人工检查，距离为显式假设；冰凌基线设为1.25m；不以长度或测距通过率拟合。原模型、旧样例均保留。紫色掩膜冲突仍需人工修正。</p>'
    for row in rows:
        stem=row['stem']; html+=f'<h2>{stem}</h2><img src="{stem}/comparison.jpg"><img src="{stem}/cloud_comparison.png"><p><a href="{stem}/pseudo_project.json">工作台可加载的项目</a></p>'
    (output/'review.html').write_text(html,encoding='utf-8')


if __name__ == '__main__':
    main()
