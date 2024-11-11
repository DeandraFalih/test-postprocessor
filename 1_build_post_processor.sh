cd "$(dirname "$0")"

python3 -m venv ~/venvs/evs
. ~/venvs/evs/bin/activate

mkdir build
cd build
cmake ..
make 

# Copy built postprocessor to the correct location
sudo mkdir -p /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors
sudo cp -f ../postprocessor-python-example/external_postprocessors.json /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
sudo cp -f ./postprocessor-python-example/postprocessor-python-example /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
sudo chmod -R 777 /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
