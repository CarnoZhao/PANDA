for bs in 16 32; do
for lr in 5e-3 1e-3 5e-4 1e-4; do
for loss in cross focal; do
for ep in 20 30; do
wd=1e-2
python IafossTrainner.py -s 0.01 -d 3 -N eff0 -e $ep -l $lr -L $loss -w $wd -b $bs -o over &
sleep 60
wd=1e-3
python IafossTrainner.py -s 0.01 -d 3 -N eff0 -e $ep -l $lr -L $loss -w $wd -b $bs -o over
done; done; done; done