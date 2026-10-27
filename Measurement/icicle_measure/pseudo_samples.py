"""Generate the six approved review samples, never the entire RGB directory."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .pseudo_depth import (ROOT, DEFAULT_RGB_DIR, DEFAULT_LABEL_DIR, SAMPLE_STEMS,
                           PseudoConfig, load_project)
from .pseudo_gui import PseudoDepthDialog, scene_preview
from .io import write_image, write_json, read_json


def make_review(directory):
    root = Path(directory)
    labels = [('rgb', '原 RGB 与预测'), ('masks', 'MobileSAM 掩膜 / 重叠待修'),
              ('background', '全图模型背景映射'), ('complete', '完整假想表面'),
              ('observed', '带误差观测深度'), ('uncertainty', '来源与不确定区域')]
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    font = ImageFont.truetype(font_path, 17)
    small_font = ImageFont.truetype(font_path, 13)
    html = ['<!doctype html><meta charset="utf-8"><title>冰凌伪深度样例</title>',
        '<style>body{background:#101c2b;color:#dce6ef;font:16px system-ui;margin:24px} '
        '.warning{padding:16px;background:#493d1e;color:#ffe2a1} '
        '.grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:14px}'
        'figure{margin:0;background:#1a2a3d;padding:12px;border-radius:8px} '
        'img{width:100%;cursor:zoom-in}a{color:#83caff} '
        'table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:6px;border-bottom:1px solid #324359}'
        'summary{cursor:pointer;font-size:21px;padding:12px}details{margin:20px 0} '
        '.bar{height:16px;background:linear-gradient(90deg,#30123b,#4687fa,#1ae4b6,#a4fc3c,#f9b938,#d73505,#7a0403)}</style>',
        '<h1>冰凌场景伪深度 · 六张样例</h1>',
        '<p class="warning">模型伪深度 / 人工合成，非相机实测。距离先验 0.5–2 m；长度常见 18–22 cm、极限 17–25 cm，仅提示；'
        '固定色标 0.5–4 m，黑色无效。所有图采用同一显示范围，图像亮暗不代表精度。</p>',
        '<div class="bar"></div><p>近 0.5 m ←　固定米制色标　→ 远 4 m；超范围截色，原数据保留。</p>',
        '<p>点击图片可看原尺寸。打开工作台右上角“RGB 生成伪深度”，可切换样例、编辑掩膜，送入主界面查看端点分布和可旋转点云。</p>']
    strips = []
    for stem in SAMPLE_STEMS:
        folder = root / stem
        if not (folder / 'pseudo_project.json').exists():
            continue
        result = load_project(folder)
        html.append('<details open><summary>' + stem + '</summary><div class="grid">')
        grid = Image.new('RGB', (1280, 3 * 390 + 50), '#101c2b')
        draw = ImageDraw.Draw(grid)
        draw.text((12, 10), stem + ' · 模型伪深度 / 人工合成 · 固定色标 0.5–4 m', font=font, fill='#ffe2a1')
        for index, (kind, title) in enumerate(labels):
            image = scene_preview(result, kind)
            write_image(folder / ('review_' + kind + '.png'), image)
            relative = stem + '/review_' + kind + '.png'
            html.append('<figure><figcaption>' + title + '</figcaption><a href="' + relative + '"><img src="' + relative + '"></a></figure>')
            thumb = Image.fromarray(image).resize((630, 354))
            col, row = index % 2, index // 2
            x, y = col * 640, 50 + row * 390
            draw.text((x + 8, y + 2), title, font=font, fill='#dce6ef')
            grid.paste(thumb, (x + 5, y + 29))
        grid.save(folder / 'overview.jpg', quality=94)
        strips.append(np.asarray(grid.resize((640, 610))))
        html.append('</div><p>蓝=假想表面，橙=掩膜待修，紫=重叠，白=空洞，黄=背景回波，红=边界错误。</p>')
        report = read_json(folder / 'timing_and_measurement.json')
        outcomes = {row['id']: row for row in report['measurement']['observed']}
        html.append('<table><tr><th>实例</th><th>深度基线/来源</th><th>轮廓弦长提示</th><th>观测测距</th><th>提示</th></tr>')
        for row in result.report['objects']:
            outcome = outcomes.get(row['id'], {})
            length = row.get('silhouette_chord_m')
            measurement = outcome.get('distance_m')
            html.append('<tr><td>#%s</td><td>%s / %s</td><td>%s</td><td>%s / %s</td><td>%s</td></tr>' % (
                row['id'], '%.3f m' % row['base_m'] if 'base_m' in row else '—', row.get('base_source', row['status']),
                '%.1f cm' % (length * 100) if length is not None else '—', outcome.get('status', '—'),
                '%.1f mm' % (measurement * 1000) if measurement is not None else '无有效长度',
                '；'.join(row.get('warnings', [row.get('reason', '')]))))
        html.append('</table><p><a href="' + stem + '/input.json">Z16 输入清单</a> · <a href="' + stem
                    + '/pseudo_project.json">可编辑项目及参数</a> · <a href="' + stem
                    + '/measurement_observed/measurement.json">完整测距判定</a></p></details>')
    (root / 'review.html').write_text('\n'.join(html), encoding='utf-8')
    if strips:
        contact = np.zeros((610 * ((len(strips) + 1) // 2), 1280, 3), np.uint8)
        for i, strip in enumerate(strips):
            contact[(i // 2) * 610:(i // 2 + 1) * 610, (i % 2) * 640:(i % 2 + 1) * 640] = strip
        Image.fromarray(contact).save(root / 'six_samples.jpg', quality=94)
    return root / 'review.html'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-only', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'outputs' / 'pseudo_depth_samples'
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.review_only:
        for stem in SAMPLE_STEMS:
            print('SAMPLE', stem, flush=True)
            try:
                payload = PseudoDepthDialog.build_one(DEFAULT_RGB_DIR / (stem + '.jpg'),
                    DEFAULT_LABEL_DIR / (stem + '.txt'), PseudoConfig(),
                    lambda value: print(value.get('message', str(value)), flush=True), lambda: False)
                rows.append({'image': stem, 'seconds_including_io': payload['seconds'],
                    'uncached_model_seconds': payload['result'].scene.model_info['seconds'], 'status': 'complete'})
            except Exception as exc:
                rows.append({'image': stem, 'status': 'failed', 'reason': str(exc)})
                print('FAILED', str(exc), flush=True)
        successes = [row for row in rows if row['status'] == 'complete']
        report = {'samples': rows, 'device': 'cpu', 'threads': 4,
            'all_six_completed': len(successes) == 6,
            'estimated_400_model_minutes': sum(r['uncached_model_seconds'] for r in successes) / max(len(successes), 1) * 400 / 60,
            'estimate_limitations': 'Six samples only; excludes storage variation and manual corrections; bulk 400 not run.'}
        write_json(output / 'batch_report.json', report)
    print('REVIEW', make_review(output), flush=True)


if __name__ == '__main__':
    main()
