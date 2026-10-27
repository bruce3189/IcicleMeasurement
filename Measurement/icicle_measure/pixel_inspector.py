"""Read actual pixel arrays rather than recovering depth from preview colours."""
import numpy as np


def inspect_pixel(result, u, v, layer='observed', model_depth=None):
    s, cfg = result.scene, result.config
    h, w = s.valid.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    q = float(s.relative[v, u])
    complete = float(result.clean_frame.depth_m[v, u])
    observed = float(result.frame.depth_m[v, u])
    raw_complete, raw_observed = int(result.clean_raw[v, u]), int(result.observed_raw[v, u])
    # The relative model's scale is undefined; never backproject it as metres.
    arrays = {'observed': result.frame.depth_m, 'complete': result.clean_frame.depth_m,
              'background': result.background_m, 'model': model_depth}
    if layer not in arrays and layer != 'relative':
        raise ValueError('未知深度读数图层')
    depth = None if layer == 'relative' else float(arrays[layer][v, u])
    valid = bool(s.valid[v, u] and depth is not None and np.isfinite(depth) and depth > 0)
    k = s.intrinsics
    xyz = ((u-k.cx)*depth/k.fx, (v-k.cy)*depth/k.fy, depth) if valid else None
    tx, ty = s.transform['pixel_center_translation']; sx, sy = s.transform['scale_xy']
    original = ((u-tx)/sx, (v-ty)/sy) if s.valid[v,u] else None
    index = int(result.instance_map[v,u])
    instance = s.detections[index-1].id if 1 <= index <= len(s.detections) else None
    flags = int(result.uncertainty[v,u])
    reasons = [text for bit, text in [(1,'留边或模型无效'),(2,'未经实测的假想表面'),
        (4,'掩膜待修'),(8,'实例重叠未定'),(16,'观测空洞'),(32,'背景回波'),(64,'边界错误')] if flags & bit]
    if layer in ('model','relative'):
        source = '模型原始相对深度' if layer == 'relative' else '模型映射 / 人工尺度'
    else:
        source = {0:'无效',1:'模型背景 / 人工尺度',2:'假想冰凌表面',3:'人工修正'}.get(int(result.source_map[v,u]),'未知')
        if layer == 'background':
            source = '背景层 / 不含假想冰凌替换'
        elif layer == 'observed':
            if flags & 16:
                source = '观测空洞'
            elif flags & 32:
                source = '背景回波（冰凌掩膜内）'
            elif flags & 64:
                source = '边界误差 / 背景回波'
    return dict(pixel=(int(u),int(v)), original_pixel=original, layer=layer, depth_m=depth,
        valid=valid, xyz_m=xyz, instance=instance, relative_value=q, source=source,
        complete_m=complete, observed_m=observed, raw_complete=raw_complete, raw_observed=raw_observed,
        current_raw=raw_complete if layer=='complete' else raw_observed if layer=='observed' else None,
        flags=flags, reasons=reasons, content_valid=bool(s.valid[v,u]), scale=cfg.depth_scale_m_per_unit,
        rgb=tuple(int(x) for x in s.rgb[v,u]), intrinsics_source=cfg.intrinsics_source)


def pixel_text(info):
    u,v = info['pixel']
    original = info['original_pixel']
    original_text = '留边' if original is None else '(%.2f, %.2f)' % original
    depth = info['depth_m']
    if info['layer']=='relative':
        main = '相对值 %.6g（无米制单位）' % info['relative_value']
    elif info['valid']:
        main = 'Z %.5f m / %.2f mm' % (depth,depth*1000)
    else:
        main = '无效深度（不生成 XYZ）'
    raw = 'Z16 %d' % info['current_raw'] if info['current_raw'] is not None else '此层无原始 Z16'
    xyz = 'XYZ —' if info['xyz_m'] is None else 'XYZ (m) = (%+.5f, %+.5f, %.5f)' % info['xyz_m']
    target = '#'+info['instance'] if info['instance'] else '背景 / 未确认实例'
    compact = '像素 (%d, %d) · %s · %s\n%s · %s · %s' % (u,v,main,raw,xyz,target,info['source'])
    details = [f'输出像素 (u, v)：({u}, {v})', '原 RGB 像素：'+original_text,
        '当前图层：'+info['layer'], main, raw, xyz,
        '坐标系：相机 X 向右，Y 向下，Z 向前', '内参来源：'+info['intrinsics_source'],
        '实例掩膜关联：'+target, '生成来源：'+info['source'], 'RGB：'+str(info['rgb']),
        '完整表面：%.5f m；Z16=%d' % (info['complete_m'],info['raw_complete']),
        '观测深度：%.5f m；Z16=%d' % (info['observed_m'],info['raw_observed']),
        'depth_scale：%.6g m/单位' % info['scale'],
        '模型相对值：%.6g（非米）' % info['relative_value'],
        '观测缺陷 / 场景标记：'+('；'.join(info['reasons']) or '无额外标记'),
        '标记对应已生成场景；完整表面自身不含观测误差。',
        '模型伪深度 / 人工合成，数值不代表实测精度。']
    return compact, '\n\n'.join(details)
