
_clip_model, _clip_preproc = None, None

def get_clip():
    global _clip_model, _clip_preproc
    if _clip_model is None:
        print(f'  Loading CLIP {CLIP_MODEL}...')
        _clip_model, _clip_preproc = clip.load(CLIP_MODEL, device=DEVICE)
        _clip_model.eval()
        print('  CLIP loaded.')
    return _clip_model, _clip_preproc

def cache_path(key):
    h = hashlib.md5(key.encode()).hexdigest()
    return Path(CACHE_DIR) / f'{h}.npy'

def load_image_url(url, timeout=8):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert('RGB')
    except Exception:
        return None

def blank_image():
    return Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))


@torch.no_grad()
def embed_texts(texts: List[str]) -> np.ndarray:
    model, _ = get_clip()
    results  = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)
    uncached_idx, uncached_texts = [], []

    for i, t in enumerate(texts):
        cp = cache_path('t_' + str(t)[:200])
        if cp.exists():
            results[i] = np.load(cp)
        else:
            uncached_idx.append(i)
            uncached_texts.append(t)

    for start in range(0, len(uncached_texts), CLIP_BATCH):
        batch_t   = uncached_texts[start:start+CLIP_BATCH]
        batch_idx = uncached_idx[start:start+CLIP_BATCH]
        tokens    = clip.tokenize([t[:300] for t in batch_t], truncate=True).to(DEVICE)
        feats     = model.encode_text(tokens).float()
        feats     = F.normalize(feats, dim=-1).cpu().numpy()
        for j, (orig_i, text) in enumerate(zip(batch_idx, batch_t)):
            results[orig_i] = feats[j]
            np.save(cache_path('t_' + str(text)[:200]), feats[j])

    return results


@torch.no_grad()
def embed_images(urls: List[str]):
    model, preproc = get_clip()
    results   = np.zeros((len(urls), EMBED_DIM), dtype=np.float32)
    is_missing = np.zeros(len(urls), dtype=bool)
    uncached_idx, uncached_urls = [], []

    for i, url in enumerate(urls):
        cp = cache_path('i_' + str(url))
        if cp.exists():
            results[i] = np.load(cp)
        else:
            uncached_idx.append(i)
            uncached_urls.append(url)

    for start in range(0, len(uncached_urls), CLIP_BATCH):
        batch_urls = uncached_urls[start:start+CLIP_BATCH]
        batch_idx  = uncached_idx[start:start+CLIP_BATCH]
        pil_imgs, missing = [], []
        for url in batch_urls:
            img = load_image_url(url) if (isinstance(url,str) and url.startswith('http')) else None
            if img is None:
                pil_imgs.append(blank_image()); missing.append(True)
            else:
                pil_imgs.append(img); missing.append(False)

        tensors = torch.stack([preproc(im) for im in pil_imgs]).to(DEVICE)
        feats   = model.encode_image(tensors).float()
        feats   = F.normalize(feats, dim=-1).cpu().numpy()

        for j, (orig_i, url, m) in enumerate(zip(batch_idx, batch_urls, missing)):
            if m:
                is_missing[orig_i] = True
                results[orig_i]    = np.zeros(EMBED_DIM)
            else:
                results[orig_i] = feats[j]
                np.save(cache_path('i_' + str(url)), feats[j])

    return results, is_missing


def build_embeddings(df):
    texts = df['combined_text_clean'].fillna('').tolist()
    print(f'  Embedding {len(texts)} texts...')
    t_embs = embed_texts(texts)

    if USE_IMAGES and 'image_link' in df.columns:
        urls = df['image_link'].fillna('').tolist()
        print(f'  Embedding {len(urls)} images (may take a while)...')
        i_embs, missing = embed_images(urls)
    else:
        i_embs  = np.zeros((len(df), EMBED_DIM), dtype=np.float32)
        missing = np.ones(len(df), dtype=bool)

   
    sim = np.einsum('ij,ij->i', t_embs, i_embs).astype(np.float32)

    t_cols = [f'te_{i}' for i in range(EMBED_DIM)]
    i_cols = [f'ie_{i}' for i in range(EMBED_DIM)]
    df_t   = pd.DataFrame(t_embs, index=df.index, columns=t_cols)
    df_i   = pd.DataFrame(i_embs, index=df.index, columns=i_cols)
    df     = df.copy()
    df['img_missing']  = missing.astype(int)
    df['text_img_sim'] = sim
    df = pd.concat([df, df_t, df_i], axis=1)
    return df

print('CLIP embedding functions defined.')
