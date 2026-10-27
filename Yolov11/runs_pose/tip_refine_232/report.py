"""汇总原模型/等权对照/tip加权结果，并生成固定选图的三联对照。"""
import argparse
import json
from pathlib import Path
from evaluate import HOME, compare_images


def load(folder):
    return json.loads((Path(folder)/'summary.json').read_text(encoding='utf8'))


def qualifies(candidate, reference):
    """试运行门槛，非泛化保证。不能以降低检出率换取漂亮的匹配点误差。"""
    return dict(
        tip_median_15pct_better=candidate['tip']['median'] <= reference['tip']['median']*.85,
        tip_p90_not_worse=candidate['tip']['p90'] <= reference['tip']['p90'],
        tip_success_all_gt_better=candidate['tip_success10_all_gt'] > reference['tip_success10_all_gt'],
        matched_not_less=candidate['counts']['matches'] >= reference['counts']['matches'],
        extra_at_most_plus2=candidate['extra'] <= reference['extra']+2,
        box_mean_iou_drop_at_most_001=candidate['iou']['mean'] >= reference['iou']['mean']-.01,
        box_bottom_p90_at_most_plus10pct=candidate['bottom_error']['p90'] <= reference['bottom_error']['p90']*1.1,
        left_base_mean_at_most_plus15pct=candidate['base_left']['mean'] <= reference['base_left']['mean']*1.15,
        right_base_mean_at_most_plus15pct=candidate['base_right']['mean'] <= reference['base_right']['mean']*1.15)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,default=HOME/'experiments/control_tip1_b1_10ep/raw_val')
    p.add_argument('--candidate',type=Path,default=HOME/'experiments/tip3_b1_10ep/raw_val')
    p.add_argument('--name',default='pilot_comparison')
    p.add_argument('--control-label',default='A 等权微调')
    p.add_argument('--candidate-label',default='B tip加权')
    args=p.parse_args()
    source=HOME/'original_val'
    entries=[('原模型',load(source)),(args.control_label,load(args.control)),(args.candidate_label,load(args.candidate))]
    gate=qualifies(entries[-1][1],entries[0][1])
    out=HOME/args.name;out.mkdir(exist_ok=True)
    lines=['# tip加权试运行结果','',
        '同一开发val集20张图，132个GT。1280、conf=.25、NMS=.7、匹配IoU≥.5；全部为原始二维点。',
        '当前数据含同视频相邻帧，微调起点已接触相关场景；这些数值用于开发，不代表独立泛化。','',
        f'| 指标 | 原模型 | {args.control_label} | {args.candidate_label} |','|---|---:|---:|---:|']
    fields=[('匹配GT数',lambda s:s['counts']['matches'],'int'),('多余预测框',lambda s:s['extra'],'int'),
            ('tip中位误差(px)',lambda s:s['tip']['median'],'float'),('tip P90(px)',lambda s:s['tip']['p90'],'float'),
            ('全部GT tip成功率@10%',lambda s:s['tip_success10_all_gt'],'pct'),
            ('框IoU均值',lambda s:s['iou']['mean'],'float'),('框底误差P90(px)',lambda s:s['bottom_error']['p90'],'float'),
            ('左根部均值误差(px)',lambda s:s['base_left']['mean'],'float'),('右根部均值误差(px)',lambda s:s['base_right']['mean'],'float'),
            ('预测tip出框率',lambda s:s['tip_outside_rate'],'pct')]
    for label,fn,fmt in fields:
        values=[fn(s) for _,s in entries]
        cells=[f'{v*100:.1f}%' if fmt=='pct' else str(v) if fmt=='int' else f'{v:.3f}' for v in values]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','## 是否达到本轮试运行门槛','',
            '通过，可供人工复核后试用。' if all(gate.values()) else '未全部通过，不替换原模型。需要根据失败项继续调整。','']
    lines += [f'- {name}: {"通过" if ok else "未通过"}' for name,ok in gate.items()]
    lines += ['','## 长短目标','', '| GT框高 | 原tip中位误差 | A | B |','|---|---:|---:|---:|']
    for key,label in [('0','<700'),('700','700–1200'),('1200','≥1200')]:
        lines.append('| '+label+' | '+' | '.join(str(s['height_bins'][key]['median']) for _,s in entries)+' |')
    compare_images(source/'predictions.json',args.candidate/'predictions.json',out/'images')
    lines += ['','## 人工对照','',
              '按GT最长目标排序选前8张，不按改善程度挑图。每张图左=人工标注，中=原模型，右=候选原始输出；黄色点为tip。',
              '请用文件名和实例编号指出：框底应该到哪里、哪个尖端属于哪根、是否多框或漏根。','']
    for ip in sorted((out/'images').glob('*.jpg')):
        lines += [f'### {ip.stem}','',f'![{ip.stem}](images/{ip.name})','']
    (out/'结果报告.md').write_text('\n'.join(lines),encoding='utf8')
    (out/'acceptance.json').write_text(json.dumps(gate,indent=2),encoding='utf8')
    print(out/'结果报告.md')
    print(json.dumps(gate,ensure_ascii=False))


if __name__=='__main__':main()
