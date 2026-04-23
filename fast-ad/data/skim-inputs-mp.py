# skim relevant information out of ROOT files
#
# Runs on CERN EOS


import glob
import argparse
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

import numpy as np

# scikit-hep
import awkward as ak
import uproot

import utils

# type 1: ZeroBias NanoAOD  (has an "Events" TTree)
def _get_arrays_type1(uproot_str: str) -> ak.Array:
    with uproot.open(uproot_str) as file:
        event_tree = file["Events"]
        arrays = event_tree.arrays(
            [
                "run", "luminosityBlock", "Regions_et", "PV_npvs",
                "L1Jet_pt", "L1Jet_eta", "L1Jet_bx",
                "L1EtSum_pt", "L1EtSum_etSumType", "L1EtSum_bx",
            ],
            library="ak",
            how=dict,
        )

    et_array = ak.to_numpy(arrays.pop("Regions_et"))
    et_regions = utils.get_region_deposits_from_ntuple_et_array(et_array)

    # L1 jets at bx=0 (matching type2 behaviour)
    bx0_jets = arrays["L1Jet_pt"][arrays["L1Jet_bx"] == 0]
    bx0_eta  = arrays["L1Jet_eta"][arrays["L1Jet_bx"] == 0]

    first_jet_et  = ak.fill_none(ak.firsts(bx0_jets), -1)
    first_jet_eta = ak.fill_none(ak.firsts(bx0_eta),  -9)

    # L1 HTT: sumType==2, bx==0 (matching type2 behaviour)
    htt_mask = (arrays["L1EtSum_etSumType"] == 2) & (arrays["L1EtSum_bx"] == 0)
    ht = ak.fill_none(ak.firsts(arrays["L1EtSum_pt"][htt_mask]), -1)

    return ak.zip({
        "run":           arrays["run"],
        "lumi":          arrays["luminosityBlock"],
        "nPV":           arrays["PV_npvs"],
        "first_jet_et":  first_jet_et,
        "first_jet_eta": first_jet_eta,
        "ht":            ht,
        "et_regions":    ak.from_numpy(et_regions.astype(np.int32)),
    }, depth_limit=1)


# type 2: signal L1 ntuples  (l1EventTree + l1UpgradeTree + l1CaloSummaryEmuTree)
def _get_arrays_type2(uproot_str: str) -> ak.Array:
    with uproot.open(uproot_str) as file:
        event_tree = file["l1EventTree/L1EventTree"]
        event_data = event_tree.arrays(["Event/run", "Event/lumi", "Event/nPV"], library="ak")

        upgrade_tree = file["l1UpgradeTree/L1UpgradeTree"]
        jet_et  = upgrade_tree["L1Upgrade/jetEt"].array()
        jet_eta = upgrade_tree["L1Upgrade/jetEta"].array()
        sum_type = upgrade_tree["L1Upgrade/sumType"].array()
        sum_et   = upgrade_tree["L1Upgrade/sumEt"].array()
        sum_bx   = upgrade_tree["L1Upgrade/sumBx"].array()

        summary_tree = file["l1CaloSummaryEmuTree/L1CaloSummaryTree"]
        calo_data = summary_tree.arrays(["modelInput[18][14]"], library="ak")

    et_regions = ak.to_numpy(calo_data["modelInput[18][14]"])

    first_jet_et  = ak.fill_none(ak.firsts(jet_et),  -1)
    first_jet_eta = ak.fill_none(ak.firsts(jet_eta), -9)

    # L1 HTT: sumType==2, bx==0
    htt_mask = (sum_type == 2) & (sum_bx == 0)
    ht = ak.fill_none(ak.firsts(sum_et[htt_mask]), -1)

    return ak.zip({
        "run":           event_data["Event/run"],
        "lumi":          event_data["Event/lumi"],
        "nPV":           event_data["Event/nPV"],
        "first_jet_et":  first_jet_et,
        "first_jet_eta": first_jet_eta,
        "ht":            ht,
        "et_regions":    ak.from_numpy(et_regions.astype(np.int32)),
    }, depth_limit=1)


_PROCESSORS = {
    "type1": _get_arrays_type1,
    "type2": _get_arrays_type2,
}

def get_arrays(uproot_str: str, file_type: str) -> ak.Array:
    if file_type == "auto":
        with uproot.open(uproot_str) as f:
            file_type = "type1" if "Events" in f else "type2"
    return _PROCESSORS[file_type](uproot_str)

def main(args):
    uproot_strs = []
    for pattern in args.input:
        uproot_strs.extend(glob.glob(pattern))
    print(f"Found {len(uproot_strs)} files matching {args.input}")

    with uproot.recreate(args.output) as output:
        with Pool(processes=8) as pool:
            func = partial(get_arrays, file_type=args.file_type)
            pbar = tqdm(total=len(uproot_strs))

            for i, chunk in enumerate(pool.imap(func, uproot_strs)):
                if i == 0:
                    output["Events"] = chunk
                else:
                    output["Events"].extend(chunk)
                pbar.update(1)
            pbar.close()

    print(f"Successfully saved to {args.output}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Apply preselection to ROOT files")
    argparser.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=["/eos/cms/store/group/phys_exotica/axol1tl/CICADANtuples/ZeroBias/*/*/0000/output_*.root"],
        help="Input ROOT file glob pattern(s); multiple patterns may be given",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="skimmed_data.root",
        help="Output file to save the skimmed data",
    )
    argparser.add_argument(
        "--file-type",
        type=str,
        choices=["type1", "type2", "auto"],
        default="auto",
        help="File format: 'type1' (NanoAOD Events tree), 'type2' (L1 ntuple trees), or 'auto' to detect per file (default)",
    )
    main(argparser.parse_args())
