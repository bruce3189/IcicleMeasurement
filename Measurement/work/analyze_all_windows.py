"""Usage: python work/analyze_all_windows.py input.json output_directory"""
from pathlib import Path
import argparse,sys,json,gzip
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from icicle_measure.io import load_bundle
from icicle_measure.window_analysis import analyze_frame_windows

def main():
    parser=argparse.ArgumentParser(description='全窗口候选分析；诊断记录，不评分、不输出长度')
    parser.add_argument('input_bundle');parser.add_argument('output_directory')
    args=parser.parse_args()
    frame,detections,_=load_bundle(args.input_bundle)
    result=analyze_frame_windows(frame,detections)
    result['source_bundle']=str(Path(args.input_bundle).resolve())
    out=Path(args.output_directory);out.mkdir(parents=True,exist_ok=True)
    with gzip.open(out/'all_windows.json.gz','wt',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,allow_nan=False)
    lines=['# 全窗口指标表','','独立全窗诊断；不评分，不产生最佳窗、端点深度或测长输出。',
        '','|实例|端点|半径px|窗口|基础近/远|基础S|实际近/远|实际S|近簇均值m|近/远标准差m|有效/无效/MAD剔除|均值变化m|候选|说明|',
        '|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|']
    fmt=lambda x:'—' if x is None else '%.5f'%x
    for obj in result['objects']:
        for ep in obj['endpoints']:
            for r in ep['windows']:
                b=r['basic'];s=r['selected'] or {}
                lines.append('|%s|%s|%d|%d×%d|%d/%d|%s|%s/%s|%s|%s|%s/%s|%d/%d/%d|%s|%s|%s|' %
                    (obj['id'],ep['name'],r['radius_px'],r['width_px'],r['height_px'],b['near_n'],b['far_n'],fmt(b['S']),s.get('near_n','—'),s.get('far_n','—'),fmt(s.get('S')),fmt(s.get('near_mean_m')),fmt(s.get('near_std_m')),fmt(s.get('far_std_m')),r['valid_n'],r['invalid_n'],r['mad_removed_n'],fmt(r['mean_change_m']),'是' if r['candidate'] else '否','；'.join(r['reasons'])))
    (out/'all_windows.md').write_text('\n'.join(lines),encoding='utf-8')
    print('全窗记录已导出；仅供诊断，未评分、选窗或计算长度。')

if __name__=='__main__':main()
