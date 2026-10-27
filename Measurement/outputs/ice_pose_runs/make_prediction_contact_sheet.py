from pathlib import Path
from PIL import Image, ImageDraw

folder = Path(r'C:\Users\Administrator\Documents\Codex\2026-10-24\continuing-from-research-the-user-s\outputs\ice_pose_runs\predict_images_400_202610242')
files = sorted(folder.glob('*.jpg'))
indices = [round(i * (len(files) - 1) / 7) for i in range(8)]
selected = [files[i] for i in indices]
cell_w, image_h, title_h = 400, 840, 28
cols, rows = 4, 2
sheet = Image.new('RGB', (cols * cell_w, rows * (image_h + title_h)), 'white')
draw = ImageDraw.Draw(sheet)
for n, path in enumerate(selected):
    im = Image.open(path).convert('RGB')
    im.thumbnail((cell_w - 20, image_h), Image.Resampling.LANCZOS)
    x = (n % cols) * cell_w + (cell_w - im.width) // 2
    y0 = (n // cols) * (image_h + title_h)
    draw.text(((n % cols) * cell_w + 8, y0 + 5), path.name, fill='black')
    sheet.paste(im, (x, y0 + title_h))
out = folder / 'preview_contact_sheet.jpg'
sheet.save(out, quality=92)
print(out)
print('samples:', ', '.join(p.name for p in selected))
