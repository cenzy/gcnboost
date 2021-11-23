
echo 'START EXPERIMENTS'  
echo 'TOTAL EXPERIMENTS: 4'
n=1

for aggr in "sum" "mean"
do
    for operator in "GCNConv" "SAGEConv"
    do
        echo Experiment number $n with $aggr $operator
        python3 main.py --exp HomoMultiTask --type homo --mode multi_task --epochs 1000 --lr 0.001 --hidden 16 --nlayers 1 --operator $operator --aggr $aggr 
        ((n=n+1))
    done
done


#for aggr in "sum" "mean"
#do
#    for operator in "SAGEConv" "GraphConv" "GATConv"
#    do
#        for nlayer in 2 3
#        do
#            echo Experiment number $n with $aggr $operator $nlayer
#            ((n=n+1))
#            python3 main.py --exp HeteroMultiTask --type hetero --mode multi_task --epochs 200 --lr 0.001 --hidden 16 --nlayers $nlayer --operator $operator --aggr $aggr --skip 
#        done
#    done
#done


echo 'END EXPERIMENTS'  
