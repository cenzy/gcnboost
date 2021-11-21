
echo 'START EXPERIMENTS'  
echo 'TOTAL EXPERIMENTS: 36'
n=1

for label in "artist" "style" "genre"
do
    for aggr in "sum" "mean"
    do
        for operator in "GCNConv" "SAGEConv"
        do
            echo Experiment number $n with $label $aggr $operator
            python3 main.py --exp HomoSingleTask --type homo --mode single_task --label $label --epochs 200 --lr 0.001 --hidden 16 --nlayers 1 --operator $operator --aggr $aggr 
            ((n=n+1))
        done
    done
done

for label in "artist" "style" "genre"
do
    for aggr in "sum" "mean"
    do
        for operator in "GCNConv" "SAGEConv"
        do
            for nlayer in 2 3
            do
                echo Experiment number $n with $label $aggr $operator $nlayer
                ((n=n+1))
                python3 main.py --exp HomoSingleTask --type homo --mode single_task --label $label --epochs 200 --lr 0.001 --hidden 16 --nlayers $nlayer --operator $operator --aggr $aggr --skip 
            done
        done
    done
done

echo 'END EXPERIMENTS'  
