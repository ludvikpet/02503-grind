In order to train a cascading classifier (Viola Jones AdaBoost classifier), it is necessary to first
    - pip install opencv-python==3.4

Then run preprocessing script:
    - python process_dataset.py --pos <pos_class> --neg <neg_class>

Then convert positive.dat into binary format:
    - opencv_createsamples -info <pos_class>_pos.dat -vec positive.vec -w 24 -h 24

Then make directory for model and brr brr train:
    - opencv_traincascade -data <data_out> -vec positive.vec -bg <neg_class>_neg.dat -numPos <N> -numNeg <M> -numStages <X> -w <II_width> -h <II_height>

After training, test classifier performance:
    - python VJ_performance_eval.py --classifier_dir <data_out> --input_img <img.jpg> [--w <II_width>] [--h <II_height>]
        - If at previous stage you set -w and -h to (24,24), then the last two are optional

After observing performance, what does the classifier pay attention to?:
    - mkdir haar_results
    - opencv_visualisation --image=annotated_dir/<test_window_n.jpg> --model=caltech-101/<data_out>/cascade.xml --data=<haar_results>

Data collected at:
    - https://www.pexels.com/video/a-van-driving-in-the-road-8230803/
    - https://www.pexels.com/video/video-of-a-sign-stop-in-desert-9010986/

