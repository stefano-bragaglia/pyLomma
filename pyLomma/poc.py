from random import choice

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv("../data/hetionet-v1.0-edges.csv")
    # src_id, src_kind, rel, tgt_id, tgt_kind, prov

    treats = df.loc[df.rel == 'treats']
    # print(f"{len(treats):,} relation/s read of type 'treats'")
    # print(treats.head(1))

    compounds = pd.concat([
        df.loc[df.src_kind == 'Compound'][['src_id']].drop_duplicates().rename(columns={'src_id': 'id'}),
        df.loc[df.tgt_kind == 'Compound'][['tgt_id']].drop_duplicates().rename(columns={'tgt_id': 'id'}),
    ], ignore_index=True).drop_duplicates()
    # print(f"{len(compounds):,} node/s read of type 'Compound'")
    # print(compounds.head(1))

    diseases = pd.concat([
        df.loc[df.src_kind == 'Disease'][['src_id']].drop_duplicates().rename(columns={'src_id': 'id'}),
        df.loc[df.tgt_kind == 'Disease'][['tgt_id']].drop_duplicates().rename(columns={'tgt_id': 'id'}),
    ], ignore_index=True).drop_duplicates()
    # print(f"{len(diseases):,} node/s read of type 'Disease'")
    # print(diseases.head(1))

    couples = list((x, y) for x, y in treats[['src_id', 'tgt_id']].values)
    X = pd.DataFrame([{
        "src_id": compound,
        "src_kind": "Compound",
        "rel": "treats",
        "tgt_id": disease,
        "tgt_kind": "Disease",
        "prov": "PharmacotherapyDB",
        "target": (compound, disease) in couples,
    }
        for compound in compounds.id.unique()
        for disease in diseases.id.unique()])
    y = X.pop("target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=42, shuffle=True, stratify=y)

    # print(X_train)
    # print(y_train)

    # print(X_test)
    # print(y_test)

    print(X_train.sample(), choice([False, True]))

print('Done.')
