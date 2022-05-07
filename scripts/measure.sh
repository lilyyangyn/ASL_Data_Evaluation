# python3 exact_sp.py ../tests/set_3 --sizes $N1 $N2 $N
# ../build/measure -i ../tests/set_3 -r $REPEAT_TIMES

REPEAT_TIMES=10

# table2: fix N2 and N, increment N1
N1_arr=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
table2_arr=('set_3' 'set_4' 'set_5')
N_arr=(      2       2       2)
N2_arr=(     100     2       1000)

for idx in ${!table2_arr[@]}; do
    ## set parameters value from array
    table2_entry=${table2_arr[$idx]}
    N=${N_arr[$idx]}
    N2=${N2_arr[$idx]}
    table2_head="set no.\t(N1)\t(N2)\t(N)\tflops\tcycles\tN2=$N2, N=$N"

    echo $table2_entry
    echo -e $table2_head
    
    for idx1 in ${!N1_arr[@]}; do
        N1=${N1_arr[$idx1]}
        echo -e "$idx1\t$N1\t$N2\t$N"

        # echo "$ python3 exact_sp.py ../tests/$table2_entry --sizes $N1 $N2 $N"
        python3 exact_sp.py ../tests/$table2_entry --sizes $N1 $N2 $N
    
        # echo "echo \"N1=$N1 N2=$N2 N=$N\" > $table2_entry.$idx1.txt"
        echo "N1=$N1 N2=$N2 N=$N" > $table2_entry.$idx1.txt
        
        # echo "$ ../build/measure -i ../tests/$table2_entry -r $REPEAT_TIMES >> $table2_entry.$idx1.txt"
        ../build/measure -i ../tests/$table2_entry -r $REPEAT_TIMES >> $table2_entry.$idx1.txt
    
    done
    
done



# table3: fix N1 and N2, increment N
N_arr=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)
table3_arr=('set_00' 'set_01' 'set_02')
N2_arr=(     2        2        2)
N1_arr=(     2        100      1000)

for idx in ${!table3_arr[@]}; do
    ## set parameters value from array
    table3_entry=${table3_arr[$idx]}
    N1=${N1_arr[$idx]}
    N2=${N2_arr[$idx]}
    table3_head="set no.\t(N1)\t(N2)\t(N)\tflops\tcycles\tN1=$N1, N2=$N2"

    echo $table3_entry
    echo -e $table3_head
    
    for idx1 in ${!N_arr[@]}; do
        N=${N_arr[$idx1]}
        echo -e "$idx1\t$N1\t$N2\t$N"

        # echo "$ python3 exact_sp.py ../tests/$table3_entry --sizes $N1 $N2 $N"
        python3 exact_sp.py ../tests/$table3_entry --sizes $N1 $N2 $N
        
        # echo "echo \"N1=$N1 N2=$N2 N=$N\" > $table3_entry.$idx1.txt"
        echo "N1=$N1 N2=$N2 N=$N" > $table3_entry.$idx1.txt
    
        # echo "$ ../build/measure -i ../tests/$table3_entry -r $REPEAT_TIMES >> $table3_entry.$idx1.txt"
        ../build/measure -i ../tests/$table3_entry -r $REPEAT_TIMES >> $table3_entry.$idx1.txt
    
    done
    
done

