from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt, gridspec
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem
from sklearn.manifold import TSNE
import umap
import requests
import xmltodict
from datasail.sail import datasail


"""
Installation by
mamba install -c conda-forge -c bioconda -c kalininalab datasail-lite umap requests xmltodict umap
"""

def combine_dicts(ds: List[dict]):
    """
    Function to combine a list of dictionaries into a single dictionary.
    If a key is present in multiple dictionaries, the values are added together.

    :param ds: List of dictionaries
    :return: Combined dictionary
    """
    out = {}
    for d in ds:
        for k, v in d.items():
            if k not in out:
                out[k] = v
            else:
                out[k] += v
    return out


def plot_hists(df):
    """
    Function to plot histograms of the data.

    :param df: The DataFrame with the data
    """
    matplotlib.rc('font', **{'size': 16})
    fig = plt.figure(figsize=(12, 10))
    rows, cols = 2, 2
    gs = gridspec.GridSpec(rows, cols, figure=fig)
    axs = [fig.add_subplot(gs[i, j]) for i in range(rows) for j in range(cols)]

    values = 0
    for i, metric in enumerate(["pKi", "pIC50"]):
        for j, exp in enumerate(["displacement", "antagonist activity"]):
            chembl_values = df[(df["acname"] == metric) & (df["exp_type"] == exp) & (df["source"] == "chembl")][
                "acvalue_uM"].values
            patent_values = df[(df["acname"] == metric) & (df["exp_type"] == exp) & (df["source"] == "patent")][
                "acvalue_uM"].values
            values += len(chembl_values) + len(patent_values)
            print(f"{metric}\t{exp} - chembl: {len(chembl_values)}")
            print(f"{metric}\t{exp} - patent: {len(patent_values)}")
            axs[i * cols + j].hist((chembl_values, patent_values), 101, histtype='bar', stacked=True, log=True,
                                   label=["chembl", "patent"])
            if i == 0:
                axs[j].set_title(exp)  # f"{exp} ($\Sigma = {sums[exp]}$)")
            if j == 0:
                axs[i * cols].set_ylabel(metric)  # f"{metric} ($\Sigma = {sums[metric]}$)")
            if i == 0 and j == 0:
                axs[i * cols + j].legend()
    fig.tight_layout()
    plt.savefig("histogram.png")


def tSNE_embed(ax, mols, classes, use_umap=False):
    classes = np.array(classes)
    c_masks = [classes == c for c in sorted(np.unique(classes))]
    fps = np.array([list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 2, nBits=1024)) for mol in mols])
    if use_umap:
        embeds = umap.UMAP().fit_transform(fps)
    else:
        embeds = TSNE(n_components=2, learning_rate="auto", init="random", random_state=42).fit_transform(fps)
    for i, c in enumerate(sorted(np.unique(classes))):
        ax.scatter(embeds[c_masks[i], 0], embeds[c_masks[i], 1], label=c)


def plot_splits(df):
    matplotlib.rc('font', **{'size': 20})
    fig = plt.figure(figsize=(25, 12))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    tSNE_embed(axs[0], df["smiles"], df["random_10f"])
    tSNE_embed(axs[1], df["smiles"], df["DataSAIL_10f"])
    axs[0].legend(loc="upper left")
    axs[0].set_title("Random split")
    axs[1].set_title("DataSAIL split")
    fig.tight_layout()
    plt.savefig("tSNE_splits.png")


def process_patent_measurement(mol: rdchem.Mol, i: int):
    """
    Function to process a single patent measurement.

    :param mol: The molecule object
    :param i: The index of the measurement within the molecule object from the patent SDF file
    :return: A dictionary with the processed data
    """
    # Extract properties from the molecule object
    out = {prop: [mol.GetProp(prop).split("\n")[0 if len(mol.GetProp(prop).split("\n")) == 1 else i]] for prop in
           ['aid', 'sid', 'cid', 'acname', 'acvalue_uM', 'aidname', 'cid_1', 'patentid', 'cid_2', 'inchi', 'inchikey',
            'cmpdsynonym']}
    out["smiles"] = [Chem.MolToSmiles(mol)]

    # Convert acvalue_uM to pKi or pIC50
    out["acvalue_uM"] = [6 - np.log10(float(out["acvalue_uM"][0]))]
    out["acname"] = ["p" + out["acname"][0]]

    # Determine the type of experiment based on the aidname
    if "inhibitory activity" in out["aidname"][0].lower():
        out["exp_type"] = ["inhibitory activity"]
        return out
    if "antagonist activity" in out["aidname"][0].lower():
        out["exp_type"] = ["antagonist activity"]
        return out
    if "displacement" in out["aidname"][0].lower():
        out["exp_type"] = ["displacement"]
        return out
    if "antagonist" in out["aidname"][0].lower():
        out["exp_type"] = ["antagonist"]
        return out
    return {}


def process_patent_record(c: int, mol: rdchem.Mol):
    """
    Function to process a single patent record.

    :param c: The index of the record
    :param mol: The molecule object
    :return: A dictionary with the processed data
    """
    print("Patent", c)
    if (num := len(mol.GetProp("aid").split("\n"))) > 1:
        return combine_dicts([process_patent_measurement(mol, i) for i in range(num)])
    return process_patent_measurement(mol, 0)


def process_patent_data(path: Optional[Path] = None):
    """
    Function to process patent data.

    :param path: The path to the patent data file
    :return: A DataFrame with the processed data
    """
    df = pd.DataFrame(combine_dicts(
        [process_patent_record(i, mol) for i, mol in enumerate(Chem.SDMolSupplier("MCHR1_patent.sdf"))]))
    df.dropna(subset=["acvalue_uM"], inplace=True)
    df.to_csv(path, index=False)
    return df


def process_chembl_measurement(mol: rdchem.Mol, activity: dict):
    """
    Function to process a single ChEMBL measurement.

    :param mol: The molecule object
    :param activity: The activity data (one entry fetched from ChEMBL database)
    :return: A dictionary with the processed data
    """
    # Extract properties from the molecule object
    out = {
        "aid": [activity["activity_id"]],
        "sid": [None],
        "cid": [activity["assay_chembl_id"]],
        "acname": [activity["type"]],
        "acvalue_uM": [activity["standard_value"]],
        "aidname": [activity["assay_description"]],
        "cid_1": [activity["assay_chembl_id"]],
        "patentid": [None],
        "cid_2": [activity["assay_chembl_id"]],
        "inchi": [None],
        "inchikey": [None],
        "cmpdsynonym": [None],
        "smiles": [Chem.MolToSmiles(mol)],
    }

    # Convert acvalue to uM
    if "standard_units" == "mM":
        out["acvalue_uM"] *= 10 ^ 3
    elif "standard_units" == "nM":
        out["acvalue_uM"] /= 10 ^ 3
    elif "standard_units" == "pM":
        out["acvalue_uM"] /= 10 ^ 6
    # Convert acvalue_uM to pKi or pIC50
    out["acvalue_uM"] = [6 - np.log10(float(out["acvalue_uM"][0]))]
    out["acname"] = ["p" + out["acname"][0]]

    # Determine the type of experiment based on the aidname
    if "inhibitory activity" in activity["assay_description"].lower():
        out["exp_type"] = ["inhibitory activity"]
        return out
    if "antagonist activity" in activity["assay_description"].lower():
        out["exp_type"] = ["antagonist activity"]
        return out
    if "displacement" in activity["assay_description"].lower():
        out["exp_type"] = ["displacement"]
        return out
    if "antagonist" in activity["assay_description"].lower():
        out["exp_type"] = ["antagonist"]
        return out
    return {}


def process_chembl_record(c: int, mol: rdchem.Mol):
    """
    Function to process a single ChEMBL record.

    :param c: The index of the record
    :param mol: The molecule object
    :return: A dictionary with the processed data
    """
    print("ChEMBL", c)
    api_url = "https://www.ebi.ac.uk/chembl/api/data/activity/search?q={}"
    try:
        response = requests.get(api_url.format(mol.GetProp("chembl_id")))
        data = xmltodict.parse(response.text)
        if isinstance(data["response"]["activities"]["activity"], list):
            return combine_dicts(
                [process_chembl_measurement(mol, activity) for activity in data["response"]["activities"]["activity"]])
        return process_chembl_measurement(mol, data["response"]["activities"]["activity"])
    except Exception as e:  # Mainly for HTTP-Request related issues
        print("Error:", e)
        return {}


def process_chembl_data(path: Optional[Path] = None):
    """
    Function to process ChEMBL data.

    :param path: The path to the ChEMBL data file
    :return: A DataFrame with the processed data
    """
    df = pd.DataFrame(combine_dicts([process_chembl_record(i, mol) for i, mol in enumerate(Chem.SDMolSupplier("MCHR1_chembl.sdf"))]))
    df.dropna(subset=["acvalue_uM"], inplace=True)
    df.to_csv(path, index=False)
    return df


def combine_MCHR1_data(df: pd.DataFrame = None, date_code: str = "20240430"):
    """
    Function to postprocess MCHR1 data.

    :param df:
    :param date_code:
    :return:
    """
    if df is None:
        if not (p_patent := Path(date_code + "_patent_data_RJ.csv")).exists():
            patent_df = process_patent_data(path=p_patent)
        else:
            patent_df = pd.read_csv(p_patent)
        patent_df["source"] = "patent"

        if not (p_chembl := Path(date_code + "_chembl_data_RJ.csv")).exists():
            chembl_df = process_chembl_data(path=p_chembl)
        else:
            chembl_df = pd.read_csv(p_patent)
        chembl_df["source"] = "patent"

    df.drop(df[[xt not in ["displacement", "antagonist activity"] for xt in df["exp_type"]]].index, inplace=True)
    df.drop(df[[xt not in ["pKi", "pIC50"] for xt in df["acname"]]].index, inplace=True)
    df = df.groupby(["aid", "cid", "acname", "smiles", "exp_type", "source"]).agg({"acvalue_uM": "mean"}).reset_index()
    plot_hists(df)
    # df["class"] = df["acvalue_uM"].apply(lambda x: 1 if x > 6.5 else 0)
    # determine the class if a ligand based on all acvalues and how far their mean is off the threshold
    tmp = dict(df[["smiles", "acvalue_uM"]].groupby(["smiles"]).mean().reset_index().values)
    df["class"] = df["smiles"].apply(lambda x: 1 if tmp[x] > 6.5 else 0)
    df["ID"] = [f"ID{i:05d}" for i in range(len(df))]
    df.to_csv(date_code + "_MCHR1_data_RJ.csv", index=False)
    return df


def process_MCHR1_data(df: pd.DataFrame = None, date_code: str = "20240430"):
    """

    :param df:
    :param date_code:
    :return:
    """
    if df is None:
        if not (p := Path(date_code + "_MCHR1_data_RJ.csv")).exists():
            df = combine_MCHR1_data(df, date_code)
        else:
            df = pd.read_csv(p)
    e_splits, _, _ = datasail(
        techniques=["I1e", "C1e"],
        max_sec=1000,
        splits=[1] * 10,
        names=[f"Fold_{i}" for i in range(10)],
        solver="SCIP",
        epsilon=0.2,
        delta=0.2,
        e_type="M",
        e_data={row['ID']: row['smiles'] for _, row in df.iterrows()},
        e_strat={row['ID']: row['source'] for _, row in df.iterrows()},
    )
    df["random_10f"] = df["ID"].apply(lambda x: e_splits["I1e"][0].get(x, None))
    df["DataSAIL_10f"] = df["ID"].apply(lambda x: e_splits["C1e"][0].get(x, None))
    plot_splits(df)
    df.to_csv(Path(date_code + "_MCHR1_splitted_RJ.csv"))


process_MCHR1_data()
