import json, os, urllib.request

data = json.load(open('data/vqaX_test.json'))
paths = set(v['img_path'] for v in data.values())
os.makedirs('data/val2014', exist_ok=True)
base_url = 'http://images.cocodataset.org/val2014/'

for i, path in enumerate(sorted(paths)):
    filename = os.path.basename(path)
    dest = f'data/val2014/{filename}'
    if not os.path.exists(dest):
        try:
            urllib.request.urlretrieve(base_url + filename, dest)
            print(f'[{i+1}/{len(paths)}] Downloaded {filename}')
        except Exception as e:
            print(f'Failed: {filename} — {e}')
print('Done!')
