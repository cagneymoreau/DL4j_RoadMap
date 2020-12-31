package KernelViewer;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * A kernel is a 2d matrix that does an elementwise multiplication and sums each value
 * A filter is multiple kernels that perform above, then perform elementwise addition and sums each value
 *
 *  filter depth must be 8 multiple
 *
 *
 *  The filter in your network should start with simple edge detectors and gradualy,
 *  by compounding each layer form detectors for comlicated objects
 *
 *
 */



public class KernelView {


    ComputationGraph graph;
    MultiLayerNetwork network;

    JTabbedPane tabbedPane = new JTabbedPane();
    int maxDimensions = 512; //jpanel size

    JFrame f = new JFrame();
    ArrayList<JPanel> panelList = new ArrayList<>();
    ArrayList<JLabel>  labelsList = new ArrayList<>();
    ArrayList<Java2DNativeImageLoader> loadersList = new ArrayList<>();

    ArrayList<INDArray> outputList = new ArrayList<>();
    ArrayList<ComputationGraph> upsampleList = new ArrayList<>();


    ArrayList<Integer> layersToView = new ArrayList<>(); //the integer position of the layers
    List<String> namesList; //raw names from network
    Gradient gradientVar; //gradients from network


    JLabel label = new JLabel();
    Java2DNativeImageLoader loader = new Java2DNativeImageLoader();
    INDArray output; //push the kernels here and display
    ComputationGraph upsample;


    //region ---------------  public constructors

    /**
     * Pass a network into either conrtuctor and it will automatically build a window
     * You can call update once to view a network or call repeatedly to watch it train
     *
     */

    public KernelView(MultiLayerNetwork m)
    {
        network = m;

        examineNet();

        builPanel();

        update();

        //upsample = ResizeNets.buildEnlargeNet(4, 3, 12, 24);


    }

    public KernelView(ComputationGraph g)
    {

    }

    //endregion

    //not gauranteed
    private void examineNet()
    {
        namesList = network.getLayerNames();

        for (int i = 0; i < namesList.size(); i++) {

           Layer.Type t = network.getLayer(namesList.get(i)).type();

           if (t.equals(Layer.Type.CONVOLUTIONAL)){

               if (i != namesList.size()-1) {
                   layersToView.add(i);
               }
           }
        }


    }


    private void builPanel()
    {

        tabbedPane.setBounds(0,0,maxDimensions, maxDimensions);

        for (int i = 0; i < layersToView.size(); i++) {

            JPanel p = new JPanel();
            panelList.add(p);
            JLabel l = new JLabel();
            labelsList.add(l);

            Java2DNativeImageLoader IM = new Java2DNativeImageLoader();
            loadersList.add(IM);

            p.add(l);
            tabbedPane.add(namesList.get(layersToView.get(i)), p);


        }


        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.add(tabbedPane);
        f.setSize(maxDimensions , maxDimensions);
        f.setEnabled(true);
        f.setLayout(null);
        f.setVisible(true);


    }

    private void getUpSamplingNet()
    {

    }

    //load the visuals
    public void update()
    {
        boolean buildUpList = false;
        if (upsampleList.size() ==0){
            buildUpList = true;
        }

         gradientVar = network.gradient();

         if ( gradientVar == null) return;


        for (int i = 0; i <  layersToView.size(); i++) {

            //get layer
            String val = layersToView.get(i) + "_W";
            INDArray first = gradientVar.getGradientFor(val);

            //compress eachlayer into 2d activation map
            first = first.sum(1);

            //norm between zero and 1
            Number min =  first.minNumber();
            Number max = first.maxNumber();
            first = first.sub(min).div ( (double) max - (double) min);


            //all layouts are 8 x 8
            int rows = (int) first.size(0) / 8;
            first = first.reshape(1,1,first.size(1) * rows, first.size(2) * 8);


            // TODO: 12/21/2020 After the first layer this needs to build lower layer features into higher layer
            // bring the activations into visible 255 - 0 range
            INDArray colorKernel = rgbKernel((int)first.size(0),
                    (int)first.size(1) , (int)first.size(2), (int)first.size(3));
            first.muli(colorKernel);

            //drop 4th dimension so image lader can read it
            //first = first.get(NDArrayIndex.interval(0,1), NDArrayIndex.all(),
                    //NDArrayIndex.all(), NDArrayIndex.all());

            if (buildUpList){

                upsampleList.add( ResizeNets.buildEnlargeNet(4, (int)first.size(0),
                        (int) first.size(1) * rows,(int) first.size(2) * 8));

                INDArray d = null;
                outputList.add(d);
            }

            outputList.set(i, upsampleList.get(i).outputSingle(first));

            BufferedImage bf = loadersList.get(i).asBufferedImage(outputList.get(i));

            labelsList.get(i).setIcon(new ImageIcon(bf));

        }


    }


    // a tensor with all vals at 255 B,G,R
    private INDArray rgbKernel(int batch,int depth, int height, int width)
    {
        INDArray basic = Nd4j.ones(batch,  depth,height, width);

        basic.muli(255);

        return basic;

    }

}
