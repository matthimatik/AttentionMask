Setup:

1: navigate to your dir
2: make virtualenv with python 2.7:
	- command : virtualenv -p python2 myvirt (change name myvirt to whatever you like)
3: open virtualenv (every time you work with attentionmask you need to do this command)
	- source myvirt/bin/activate
5: cd AttentionMask
6: pip install -r requirements.txt
7: cd caffe
8: make pycaffe -j6
9: cd ..
10: pip install protobuf
11: pip install ipython
12: pip install opencv-contrib-python
13: pip install pycocotools

14: mkdir data params results
15: add a *.caffemodel file to the params folder
16: link ./data/coco to .../coco 

Run:
`python2 testAttentionMask.py 0 attentionMask-8-128 --init_weights attentionmask-8-128final.caffemodel --dataset train2014 --end 5000`