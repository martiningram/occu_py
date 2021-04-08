for n_checklists in 2000 4000 8000 16000 32000 64000; do
    echo "Fitting $n_checklists"
    cur_dir="./speed_test/$n_checklists/"
    data_dir="$cur_dir/data/"
    fit_dir_python="$cur_dir/python/"
    fit_dir_unmarked="$cur_dir/unmarked/"
    mkdir -p $cur_dir
    mkdir -p $fit_dir_unmarked
    python create_unmarked_data.py $n_checklists $data_dir
    python fit_fast_occu.py $data_dir $fit_dir_python
    Rscript fit_unmarked.R $data_dir $fit_dir_unmarked
done
