# Face_Recognition
Face Recognition software using Python with OpenCV library.
Uses Tensorflow as Backend.
It uses mobilenet model for retraining the neural network for added images, but user can easily change it under ARCHITECTURE later.
Steps for the program : 

1. Test camera by using command
		
		python Test_Camera.py

2. Check whether program detects face or not
		
		python Face_Detection.py

3. For creating datasets for faces on your own, run the script
		
		python Data_Gathering_Faces.py

We are using here Image Size as 224, and fraction of model as 0.5

For getting the plots for the method, run this script

		tensorboard --logdir tf_files/training_summaries &

4. For training your dataset, run the following script
													
	 IMAGE_SIZE=224
	 ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"	
	 python -m scripts.retrain \
  	 --bottleneck_dir=tf_files/bottlenecks \
  	 --how_many_training_steps=3000 \
 	 --model_dir=tf_files/models/ \
 	 --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
 	 --output_graph=tf_files/retrained_graph.pb \
 	 --output_labels=tf_files/retrained_labels.txt \
	 --architecture="${ARCHITECTURE}" \
 	 --image_dir=tf_files/Faces_Datasets

5. Now for starting the face recognition program, type and run this in your terminal

		python -m scripts.run
