## Learn a tokenizer on one dataset and then apply it to a second dataset

<details>

<summary>Localize filenames by cluster.</summary>

```sh
case "$(uname -n)" in
    cri*)
        hm="/gpfs/data/bbj-lab/users/burkh4rt"
        ;;
    bbj-lab*)
        hm="/mnt/bbj-lab/users/burkh4rt"
        ;;
    *)
        echo "Unknown host $(uname -n)"
        ;;
esac
raw_mimic="${hm}/data-raw/mimic-2.1.0"
raw_ucmc="${hm}/data-raw/ucmc-2.1.0"
```

</details>

```sh
# collate two datasets
uv run cocoa collate \
	--raw-data-home ${raw_mimic} \
	--processed-data-home ./processed/mimic

uv run cocoa collate \
	--raw-data-home ${raw_ucmc} \
	--processed-data-home ./processed/ucmc

# learn a tokenizer on mimic
uv run cocoa tokenize \
	--processed-data-home ./processed/mimic

# apply that tokenizer to ucmc
uv run cocoa tokenize \
	--tokenizer-home ./processed/mimic/tokenizer.yaml \
	--processed-data-home ./processed/ucmc

# prepare datasets for inference
uv run cocoa winnow \
    --processed-data-home ./processed/mimic

uv run cocoa winnow \
    --processed-data-home ./processed/ucmc
```
