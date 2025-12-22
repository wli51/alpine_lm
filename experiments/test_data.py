from pathlib import Path

import pandas as pd

DATA_PATH = Path("/projects/wli19@xsede.org/PRISM_dataset/")

if not DATA_PATH.exists() or not DATA_PATH.is_dir():
    raise FileNotFoundError(
        f"The specified path does not exist or is not a directory: {DATA_PATH}"
    )

tissues = list(DATA_PATH.iterdir())

valid_tissues = []
valid_cells = []
drug_set = set()
for tissue in tissues:
    if not tissue.is_dir():
        print(f"Skipping non-directory item: {tissue}")
        continue
    valid_tissues.append(tissue)

    cells = list(tissue.iterdir())
    if not cells:
        print(f"No cell line files found in tissue directory: {tissue}")
        continue

    for file in cells:
        if file.suffix == ".csv":
            valid_cells.append(file)

        _ = pd.read_csv(file)
        print(_.head(2))
        if "ccle_name" not in _.columns:
            raise ValueError(
                f"Required columns 'ccle_name' not found in file: {file}"
            )
        if "name" not in _.columns:
            raise ValueError(
                f"Required columns 'name' not found in file: {file}"
            )
        drug_names = _.name.unique().tolist()
        drug_set.update(drug_names)
        if "ic50" in _.columns:
            if not pd.api.types.is_numeric_dtype(_["ic50"]):
                raise ValueError(
                    f"'ic50' column must be numeric in file: {file}"
                )

if not valid_tissues:
    raise ValueError("No valid tissue directories found.")
if not valid_cells:
    raise ValueError("No valid cell line CSV files found.")

print(f"Total valid tissues found: {len(valid_tissues)}")
print(f"Total valid cells found: {len(valid_cells)}")
print(f"Total unique drugs found: {len(drug_set)}")

drug_df = pd.DataFrame({"drug_name": list(drug_set)})
drug_df.to_csv("drug_set.csv", index=False)
print(f"Exported {len(drug_set)} unique drugs to drug_set.csv")
