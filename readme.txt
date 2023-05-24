1. The modules for our medium+perception-driven procedure are in ./art/
This procedure is trained in ./train.py 
To apply the trained model use render_patchwise.py 


2. The inspiration-driven procedure uses the style transfer from the sub-repository style_transfer.
To use the inspiration procedure with a hardcoded baseline technique, run render_hardcoded.py


Imagination images are made with style_transfer/styletransfer.py or ./imagine.sh
To rigorously ensure that the inspiration procedure has never seen a human artwork
we use a re-trained VGG19 network for style transfer.

We re-trained VGG19 on the iNaturalist 2021 dataset:
https://github.com/visipedia/inat_comp/tree/master/2021
to allow us to do style transfer with a network that has never seen human artworks.
Specifically we trained the network to classify the "order" (273 categories)
in the "train mini" subset (42GB) of the iNaturalist dataset