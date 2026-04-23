#!/bin/bash
# Re-skim all event types from CICADANtuples into Data/raw_root/
# Runs on CERN EOS - configure paths

BASE="/eos/cms/store/group/phys_exotica/axol1tl/CICADANtuples"
OUT="/eos/home-l/loshaugh/SWAN_projects/Thesis/Data/raw_root"
SCRIPT="/eos/home-l/loshaugh/SWAN_projects/Thesis/skim-inputs-mp.py"

run_skim() {
    # Usage: run_skim <out_file> <file_type> <glob_pattern> [<glob_pattern> ...]
    local out_file="$1"
    local file_type="$2"
    shift 2
    echo "=== Skimming -> ${out_file} ==="
    python3 "$SCRIPT" \
        --input "$@" \
        --output "${OUT}/${out_file}" \
        --file-type "${file_type}"
}

run_skim "glugluhtogg.root"       "auto"  "${BASE}/GluGluHToGG_M-125_TuneCP5_13p6TeV_powheg-pythia8/*/*/0000/output_*.root"
run_skim "glugluhtotautau.root"   "auto"  "${BASE}/GluGluHToTauTau_M-125_TuneCP5_13p6TeV_powheg-pythia8/*/*/0000/output_*.root"
run_skim "hto2longlivedto4b.root" "auto"  "${BASE}/HTo2LongLivedTo4b_MH-125_MFF-12_CTau-900mm_TuneCP5_13p6TeV_pythia8/*/*/0000/output_*.root"
run_skim "singleneutrino.root"    "auto"  "${BASE}/SingleNeutrino_E-10-gun/*/*/0000/output_*.root"
run_skim "suep.root"              "type1" "${BASE}/SUEP/*/*/0000/output_*.root"
run_skim "tt.root"                "auto"  "${BASE}/TT_TuneCP5_13p6TeV_powheg-pythia8/*/*/0000/output_*.root"
run_skim "vbfhto2b.root"          "auto"  "${BASE}/VBFHto2B_M-125_TuneCP5_13p6TeV_powheg-pythia8/*/*/0000/output_*.root"
run_skim "vbfhtotautau.root"      "auto"  "${BASE}/VBFHToTauTau_M125_TuneCP5_13p6TeV_powheg-pythia8/*/*/0000/output_*.root"
run_skim "zprimetotautau.root"    "auto"  "${BASE}/ZprimeToTauTau_M-4000_TuneCP5_tauola_13p6TeV-pythia8/*/*/0000/output_*.root"
run_skim "zz.root"                "auto"  "${BASE}/ZZ_TuneCP5_13p6TeV_pythia8/*/*/0000/output_*.root"

# ZeroBias: specific Run2024I campaigns only
run_skim "zb.root" "auto" \
    "${BASE}/ZeroBias/AnomalyDetectionPaper2025_ZeroBias_Run2024I_CERN_24Apr2025/*/0000/output_*.root" \
    "${BASE}/ZeroBias/AnomalyDetectionPaper2025_ZeroBias_Run2024I_SpecificRuns_CERN_03Apr2025/*/0000/output_*.root" \
    "${BASE}/ZeroBias/AnomalyDetectionPaper2025_ZeroBias_Run2024I_SpecificRuns_CERN_14Apr2025/*/0000/output_*.root" \
    "${BASE}/ZeroBias/AnomalyDetectionPaper2025_ZeroBias_Run2024I_SpecificRuns_CERN_16Apr2025/*/0000/output_*.root" \
    "${BASE}/ZeroBias/AnomalyDetectionPaper2025_ZeroBias_Run2024I_SpecificRuns_CERN_17Apr2025/*/0000/output_*.root" \
    "${BASE}/ZeroBias/AnomalyDetectionPaper2025_ZeroBias_Run2024I_SpecificRuns_CERN_30Apr2025/*/0000/output_*.root"

echo "=== All done ==="
