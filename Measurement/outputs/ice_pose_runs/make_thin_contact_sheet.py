from pathlib import Path
from PIL import Image, ImageDraw
root = Path(r'C:\Users\Administrator\Documents\Codex\2026-10-24\continuing-from-research-the-user-s\outputs\ice_pose_runs\predict_images_400_202610242')
folder = root / 'thin_overlay'
files = sorted(folder.glob('*.jpg'))
indices = [round(i * (len(files) - 1) / 7) for i in range(8)]
selected = [files[i] for i in indices]
cell_w, image_h, title_h = 400, 840, 28
sheet = Image.new('RGB', (4 * cell_w, 2 * (image_h + title_h)), 'white')
draw = ImageDraw.Draw(sheet)
for n, path in enumerate(selected):
    im = Image.open(path).convert('RGB')
    im.thumbnail((cell_w - 20, image_h), Image.Resampling.LANCZOS)
    x = (n % 4) * cell_w + (cell_w - im.width) // 2
    y0 = (n // 4) * (image_h + title_h)
    draw.text(((n % 4) * cell_w + 8, y0 + 5), path.name, fill='black')
    sheet.paste(im, (x, y0 + title_h))
out = root / 'preview_contact_sheet_thin.jpg'
sheet.save(out, quality=92)
print(out)
