clear

echo "Creating datasets for n clients:"

gnome-terminal -e "python data/federated_data_extractor.py"

sleep 3

echo "Start federated learning on n clients:"
gnome-terminal -e "python server.py -p 5000 -i 127.0.0.1 -u 3"

sleep 3

for i in `seq 0 1 2`;
        do
                echo "Start client $i"
                gnome-terminal -e "python client.py -s 127.0.0.1:5000 -c $i -e 5"
        done

sleep 3
