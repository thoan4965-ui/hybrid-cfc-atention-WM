import re

with open(r'C:\Users\duong\.gemini\antigravity\brain\edecb900-e272-408b-81b6-2c93b63b35ef\.system_generated\steps\86\content.md', encoding='utf-8') as f:
    content = f.read()

transmissions = re.findall(r'transmission name="([a-z_]+)_tran"', content)
for i, t in enumerate(transmissions, 1):
    print(f'{i:02}. {t}')
print(f'\nTOTAL transmissions: {len(transmissions)}')

# Phan loai
wrist = [t for t in transmissions if 'wrist' in t]
thumb = [t for t in transmissions if 'thumb' in t]
index = [t for t in transmissions if 'index' in t]
middle = [t for t in transmissions if 'middle' in t]
ring = [t for t in transmissions if 'ring' in t]
pinky = [t for t in transmissions if 'pinky' in t]

print(f'\n--- WRIST ({len(wrist)}): {wrist}')
print(f'--- THUMB ({len(thumb)}): {thumb}')
print(f'--- INDEX ({len(index)}): {index}')
print(f'--- MIDDLE ({len(middle)}): {middle}')
print(f'--- RING ({len(ring)}): {ring}')
print(f'--- PINKY ({len(pinky)}): {pinky}')

finger_total = len(thumb) + len(index) + len(middle) + len(ring) + len(pinky)
print(f'\nFingers only (no wrist): {finger_total}')
