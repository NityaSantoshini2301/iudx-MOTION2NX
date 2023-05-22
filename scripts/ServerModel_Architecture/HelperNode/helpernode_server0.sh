#! /bin/bash
# paths required to run cpp files
image_config=${BASE_DIR}/config_files/file_config_input_remote
model_config=${BASE_DIR}/config_files/file_config_model
build_path=${BASE_DIR}/build_debwithrelinfo_gcc
debug_0=${BASE_DIR}/logs/server0/
scripts_path=${BASE_DIR}/scripts
smpc_config_path=${BASE_DIR}/config_files/smpc-remote-config.json
smpc_config=`cat $smpc_config_path`
# #####################################################################################################################################
cd $build_path

if [ -f MemoryDetails0 ]; then
   rm MemoryDetails0
   # echo "Memory Details0 are removed"
fi

if [ -f AverageMemoryDetails0 ]; then
   rm AverageMemoryDetails0
   # echo "Average Memory Details0 are removed"
fi

if [ -f AverageMemory0 ]; then
   rm AverageMemory0
   # echo "Average Memory Details0 are removed"
fi

if [ -f AverageTimeDetails0 ]; then
   rm AverageTimeDetails0
   # echo "AverageTimeDetails0 is removed"
fi

if [ -f AverageTime0 ]; then
   rm AverageTime0
   # echo "AverageTime0 is removed"
fi
# #####################Inputs##########################################################################################################

# Do dns resolution or not
cs0_dns_resolve=`echo $smpc_config | jq -r .cs0_dns_resolve`
cs1_dns_resolve=`echo $smpc_config | jq -r .cs1_dns_resolve`
helpernode_dns_resolve=`echo $smpc_config | jq -r .helpernode_dns_resolve`
reverse_ssh_dns_resolve=`echo $smpc_config | jq -r .reverse_ssh_dns_resolve`

# cs0_host is the ip/domain of server0, cs1_host is the ip/domain of server1
cs0_host=`echo $smpc_config | jq -r .cs0_host`
cs1_host=`echo $smpc_config | jq -r .cs1_host`
helpernode_host=`echo $smpc_config | jq -r .helpernode_host`
reverse_ssh_host=`echo $smpc_config | jq -r .reverse_ssh_host`

if [[ $cs0_dns_resolve == "true" ]];
then
cs0_host=`dig +short $cs0_host | grep '^[.0-9]*$' | head -n 1`
fi

if [[ $cs1_dns_resolve == "true" ]];
then
cs1_host=`dig +short $cs1_host | grep '^[.0-9]*$' | head -n 1`
fi

if [[ $helpernode_dns_resolve == "true" ]];
then
helpernode_host=`dig +short $helpernode_host | grep '^[.0-9]*$' | head -n 1`
fi

if [[ $reverse_ssh_dns_resolve == "true" ]];
then 
reverse_ssh_host=`dig +short $reverse_ssh_host | grep '^[.0-9]*$' | head -n 1`
fi

# Ports on which weights,image provider  receiver listens/talks
cs0_port_model_receiver=`echo $smpc_config | jq -r .cs0_port_model_receiver`
cs0_port_image_receiver=`echo $smpc_config | jq -r .cs0_port_image_receiver`

# Ports on which Image provider listens for final inference output
cs0_port_cs0_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs0_output_receiver`
cs0_port_cs1_output_receiver=`echo $smpc_config | jq -r .cs0_port_cs1_output_receiver`

# Ports on which server0 and server1 of the inferencing tasks talk to each other
cs0_port_inference=`echo $smpc_config | jq -r .cs0_port_inference`
cs1_port_inference=`echo $smpc_config | jq -r .cs1_port_inference`
helpernode_port_inference=`echo $smpc_config | jq -r .helpernode_port_inference`
relu0_port_inference=`echo $smpc_config | jq -r .relu0_port_inference`
relu1_port_inference=`echo $smpc_config | jq -r .relu1_port_inference`


fractional_bits=`echo $smpc_config | jq -r .fractional_bits`

# Index of the image for which inferencing task is run
image_id=`echo $smpc_config | jq -r .image_id`

# echo all input variables
# echo "cs0_host $cs0_host"
# echo "cs1_host $cs1_host"
# echo "cs0_port_model_receiver $cs0_port_model_receiver"
# echo "cs0_port_image_receiver $cs0_port_image_receiver"
# echo "cs0_port_cs1_output_receiver $cs0_port_cs1_output_receiver"
# echo "cs0_port_inference $cs0_port_inference"
# echo "cs1_port_inference $cs1_port_inference"
# echo "fractional bits: $fractional_bits"
##########################################################################################################################################


if [ ! -d "$debug_0" ];
then
        # Recursively create the required directories
        mkdir -p $debug_0
fi

######################### Weights Share Receiver ############################################################################################
echo "Weight shares receiver starts"
$build_path/bin/Weights_Share_Receiver_remote --my-id 0 --port $cs0_port_model_receiver --file-names $model_config --current-path $build_path > $debug_0/Weights_Share_Receiver0.txt &
pid1=$!

#########################Image Share Receiver ############################################################################################
echo "Image shares receiver starts"
$build_path/bin/Image_Share_Receiver --my-id 0 --port $cs0_port_image_receiver --fractional-bits $fractional_bits --file-names $image_config --current-path $build_path > $debug_0/Image_Share_Receiver0.txt &
pid2=$!
wait $pid2 $pid1
echo "Weight shares received"
echo "Image shares received"
########################Share generators end ############################################################################################

######################## Inferencing task starts ###############################################################################################


echo "Inferencing task of the image shared starts"

############################Inputs for inferencing tasks #######################################################################################
layer_id=1
# input_config=" "
image_share="remote_image_shares"
if [ $layer_id -eq 1 ];
then
    input_config="remote_image_shares"
fi

####################################### Matrix multiplication layer 1 ###########################################################################

start=$(date +%s)
$build_path/bin/server0 --WB_file file_config_model0 --input_file $input_config  --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --helper_node $helpernode_host,$helpernode_port_inference --current-path $build_path --layer-id $layer_id --fractional-bits $fractional_bits > $debug_0/server0_layer${layer_id}.txt &
pid3=$!
wait $pid3
echo "Layer 1: Matrix multiplication and addition is done"

# #######################################ReLu layer 1 ####################################################################################
$build_path/bin/tensor_gt_relu --my-id 0 --party 0,$cs0_host,$relu0_port_inference --party 1,$cs1_host,$relu1_port_inference --arithmetic-protocol beavy --boolean-protocol yao --fractional-bits $fractional_bits --filepath file_config_input0 --current-path $build_path > $debug_0/tensor_gt_relu1_layer0.txt &
pid4=$!
wait $pid4

echo "Layer 1: ReLU is done"

#######################Next layer, layer 2, inputs for layer 2 ###################################################################################################
((layer_id++))

# #Updating the config file for layers 2 and above.
if [ $layer_id -gt 1 ];
then
    input_config="outputshare"
fi


#######################################Matrix multiplication layer 2 ###########################################################################
$build_path/bin/server0 --WB_file file_config_model0 --input_file $input_config  --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --helper_node $helpernode_host,$helpernode_port_inference  --current-path $build_path --layer-id $layer_id --fractional-bits $fractional_bits > $debug_0/server0_layer${layer_id}.txt &

pid5=$!
wait $pid5

echo "Layer 2: Matrix multiplication and addition is done"

####################################### Argmax  ###########################################################################
$build_path/bin/argmax --my-id 0 --threads 1 --party 0,$cs0_host,$cs0_port_inference --party 1,$cs1_host,$cs1_port_inference --arithmetic-protocol beavy --boolean-protocol beavy --repetitions 1 --config-filename file_config_input0 --config-input $image_share --current-path $build_path > $debug_0/argmax0_layer2.txt &

pid6=$!
wait $pid6
echo "Layer 2: Argmax is done"
end=$(date +%s)

####################################### Final output provider  ###########################################################################
$build_path/bin/final_output_provider --my-id 0 --connection-ip $reverse_ssh_host --connection-port $cs0_port_cs0_output_receiver --config-input $image_share --current-path $build_path > $debug_0/final_output_provider0.txt &
pid7=$!
wait $pid7
echo "Output shares of server 0 sent to the Image provider"

#To catch any extra processes
wait

awk '{ sum += $1 } END { print sum }' AverageTimeDetails0 >> AverageTime0
#  > AverageTimeDetails0 #clearing the contents of the file

sort -r -g AverageMemoryDetails0 | head  -1 >> AverageMemory0
#  > AverageMemoryDetails0 #clearing the contents of the file

echo -e "Inferencing Finished"

Mem=`cat AverageMemory0`
Time=`cat AverageTime0`

Mem=$(printf "%.2f" $Mem)
Convert_KB_to_GB=$(printf "%.14f" 9.5367431640625E-7)
Mem2=$(echo "$Convert_KB_to_GB * $Mem" | bc -l)

Memory=$(printf "%.3f" $Mem2)

echo "Memory requirement:" `printf "%.3f" $Memory` "GB"
echo "Time taken by inferencing task:" $Time "ms"
echo "Elapsed Time: $(($end-$start)) seconds"

cd $scripts_path
