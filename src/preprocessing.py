UNIT_MAP = {
    'kilogram':'kg','kilograms':'kg','kgs':'kg',
    'gram':'g','grams':'g','gm':'g','gms':'g',
    'milligram':'mg','milligrams':'mg',
    'pound':'lb','pounds':'lb','lbs':'lb',
    'ounce':'oz','ounces':'oz',
    'litre':'l','litres':'l','liter':'l','liters':'l',
    'millilitre':'ml','millilitres':'ml','milliliter':'ml',
    'fluid ounce':'fl oz','fluid ounces':'fl oz',
    'piece':'pc','pieces':'pc','pcs':'pc',
    'count':'ct','counts':'ct','pack':'pk','packs':'pk',
    'set':'set','sets':'set',
}
BRAND_KEYWORDS = [
    'samsung','apple','sony','lg','hp','dell','lenovo','asus','bosch',
    'philips','whirlpool','havells','bajaj','prestige','milton','amul',
    'nestle','britannia','patanjali','himalaya','dabur','colgate',
]
QUALITY_KEYWORDS = [
    'organic','premium','natural','original','authentic',
    'pure','fresh','raw','refined','whole','extra virgin',
]
PACK_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(kg|g|gm|gms|gram|grams|kilogram|kilograms|kgs'
    r'|ml|l|litre|litres|liter|liters|millilitre|millilitres'
    r'|lb|lbs|oz|fl\s*oz|mg|pc|pcs|piece|pieces|count|ct'
    r'|pack|packs|pk|set|sets)',
    re.IGNORECASE,
)


def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower().strip()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_brand(text):
    if not isinstance(text, str): return 'unknown'
    tl = text.lower()
    for b in BRAND_KEYWORDS:
        if b in tl: return b
    tokens = text.split()
    if tokens and tokens[0][0].isupper(): return tokens[0].lower()
    return 'unknown'

def extract_pack_size(text):
    if not isinstance(text, str): return (np.nan, 'unknown')
    m = PACK_RE.findall(text)
    if m:
        qty, unit = m[0]
        unit = UNIT_MAP.get(unit.lower(), unit.lower())
        return (float(qty), unit)
    return (np.nan, 'unknown')

def normalize_to_grams(qty, unit):
    conv = {'kg':1000,'g':1,'mg':0.001,'lb':453.59,'oz':28.35}
    if pd.isna(qty): return np.nan
    return qty * conv.get(unit, 1)


def fit_lda(texts, n_topics=N_TOPICS):
    n_docs = len(texts)
    vec = CountVectorizer(
        max_features=5000, stop_words='english',
        min_df=min(5, max(1, int(n_docs * 0.01))),
        max_df=0.95,
    )
    X = vec.fit_transform(texts.fillna(''))
    lda = LatentDirichletAllocation(
        n_components=min(n_topics, n_docs),
        random_state=SEED, max_iter=20, learning_method='online',
    )
    lda.fit(X)
    return vec, lda

def transform_lda(texts, vec, lda):
    X = vec.transform(texts.fillna(''))
    tm = lda.transform(X)
    return pd.DataFrame(tm, index=texts.index,
                        columns=[f'lda_{i}' for i in range(tm.shape[1])])



def preprocess(df, fit=True, artifacts=None):
    """
    fit=True  : fit LDA + scaler (use on train set)
    fit=False : reuse fitted artifacts (use on val/test)
    Returns   : df_out, artifacts_dict
    """
    df = df.copy()

    
    df['combined_text']       = df['catalog_content'].fillna('')
    df['combined_text_clean'] = df['combined_text'].apply(clean_text)

    
    df['text_length']       = df['combined_text'].str.len().fillna(0)
    df['word_count']        = df['combined_text'].str.split().str.len().fillna(0)
    df['num_digits']        = df['combined_text'].apply(
                                  lambda t: sum(c.isdigit() for c in str(t)))
    df['num_special']       = df['combined_text'].apply(
                                  lambda t: sum(not c.isalnum() and not c.isspace()
                                               for c in str(t)))

   
    df['brand'] = df['combined_text'].apply(extract_brand)

    
    pack = df['combined_text'].apply(lambda t: pd.Series(extract_pack_size(t)))
    df['raw_qty']  = pack[0]
    df['unit']     = pack[1]
    df['norm_qty'] = df.apply(lambda r: normalize_to_grams(r['raw_qty'], r['unit']), axis=1)


    for kw in QUALITY_KEYWORDS:
        col = 'flag_' + kw.replace(' ', '_')
        df[col] = df['combined_text_clean'].str.contains(kw, case=False, na=False).astype(int)


    if fit:
        vec, lda = fit_lda(df['combined_text_clean'])
        artifacts = artifacts or {}
        artifacts['lda_vec'] = vec
        artifacts['lda_model'] = lda
    else:
        vec  = artifacts['lda_vec']
        lda  = artifacts['lda_model']

    lda_df = transform_lda(df['combined_text_clean'], vec, lda)
    df = pd.concat([df, lda_df], axis=1)

  
    for col in ['brand', 'unit']:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            artifacts[f'le_{col}'] = le
        else:
            le = artifacts[f'le_{col}']
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])


    num_cols = ['text_length','word_count','num_digits','num_special','norm_qty']
    df[num_cols] = df[num_cols].fillna(0)

    return df, artifacts

print('Preprocessing functions defined.')
