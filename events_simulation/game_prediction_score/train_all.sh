declare -a leagues=("E0" "SP1" "I1" "D1" "F1")

for league in "${leagues[@]}"
do
    echo "Training $league..."
    python main.py $league
done
