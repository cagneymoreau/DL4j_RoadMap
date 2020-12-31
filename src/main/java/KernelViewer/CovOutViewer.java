package KernelViewer;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 *
 *
 *
 * This class is a convolutional layer viewer that will automatically fin conv layers
 * and create visual displays.
 * found this after finished
 * https://stackoverflow.com/questions/61236583/how-to-obtain-data-of-filters-of-the-convolution-layers-of-the-cnn-network-in-dl
 * https://github.com/eclipse/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-ui/src/main/java/org/deeplearning4j/ui/weights/ConvolutionalIterationListener.java
 *
 *
 * Type 1) Activation map Visuals
 *
 * This shows where each filter was activated the most.
 * The expectation is for each layer to become more and more abstract
 * The first layers filter should be simple edge and color detectors and will show
 * a higher resolution activation map where the image may even be interpretable to the naked eye
 * The later layers should show abstract blobs or no activation as the complex filters which
 * detect patterns or objects. For example in the yolo filters a picture of a person will show
 * some bright pixels over a face. However the final scissor filter should be almost all black
 *
 * Type 2) Kernel Visuals
 *
 * A kernel is a 2d matrix that does an elementwise multiplication and sums each value
 *  A filter is multiple kernels that perform above, then perform elementwise addition and sums each value
 *
 *   filter depth must be 8 multiple or filters will be dropped with a console warning
 *
 *  The filter in your network should start with simple edge detectors and gradualy,
 *  by compounding each layer form detectors for complicated objects
 *
 */



public class CovOutViewer {


    ComputationGraph graph;
    MultiLayerNetwork network;

    JTabbedPane tabbedPane = new JTabbedPane();
    int maxDimensions = 1024; //jpanel size

    JFrame f = new JFrame();
    ArrayList<JPanel> panelList = new ArrayList<>();
    ArrayList<JLabel>  labelsList = new ArrayList<>();
    ArrayList<Java2DNativeImageLoader> loadersList = new ArrayList<>();

    ArrayList<Integer> layersToView = new ArrayList<>(); //the integer position of the layers
    List<String> namesList; //raw names from network

    ArrayList<ComputationGraph> upsamples = new ArrayList<>();

    ArrayList<ComputationGraph> upsampleList = new ArrayList<>();
    Map<String, INDArray> paramVar; //param weights


    //1 = activations, 2 = kernel
    int chosenOption = 0;

    //region ---------------  public constructors

    /**
     * Pass a network into either conrtuctor and it will automatically build a window
     * You can call update once to view a network or call repeatedly to watch it train
     *
     */

    public CovOutViewer(MultiLayerNetwork m)
    {
        network = m;

        //chosenOption = option;

        examineNet();

        builPanel();

    }

    public CovOutViewer(ComputationGraph g)
    {
        graph = g;

        //chosenOption = option;

        examineGraph();


        builPanel();
    }

    //endregion


    //region -------------- multilayer net
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


     //load the activation visuals
     public void update(List<INDArray> a)
     {

                if (a == null) return;

                int mirrorIssue = 1;
                for (int i = 0; i <  layersToView.size(); i++) {

                    if (i == layersToView.size()/2 -1){
                        mirrorIssue = 0;
                    }


                    //get layer

                    INDArray workArr = a.get(i+ mirrorIssue);

                    updateResults(workArr, i);

                }


            }


    //endregion


    //region-------------- computationalgraph

            private void examineGraph()
            {
                Map<String, GraphVertex> mMap =null;
                try {
                    Class myclass = graph.getClass();
                    Field f = myclass.getDeclaredField("verticesMap");
                    f.setAccessible(true);

                    mMap = (Map<String, GraphVertex>) f.get(graph);
                }catch (Exception e){
                    e.printStackTrace();
                    return;
                }

                namesList = new ArrayList<>(mMap.keySet());


                for (int i = 0; i < namesList.size(); i++) {

                    try {
                    Layer.Type t = graph.getLayer(namesList.get(i)).type();


                        if (t.equals(Layer.Type.CONVOLUTIONAL)) {

                            if (i != namesList.size() - 1) {
                                layersToView.add(i);
                            }
                        }
                    }catch (Exception e){
                        e.printStackTrace();
                    }

                }


            }


            //load the visuals
            public void update(Map<String, INDArray> output)
            {

                if (output == null) return;

                int mirrorIssue = 1;
                for (int i = 0; i <  layersToView.size(); i++) {

                    if (i == layersToView.size()/2 -1){
                        mirrorIssue = 0;
                    }

                    //get layer
                    String name = "";
                    try {
                         name = namesList.get(layersToView.get(i));

                        INDArray workArr = output.get(name);

                        updateResults(workArr, i);
                    }catch (Exception e){
                        System.out.println("Position: " + i + "  Name: " + name);
                        e.printStackTrace();
                    }

                }


            }



            //endregion


    //region----------- helpers/generic



    public void networkReverse(int chann, int width, int height, boolean norm, INDArray out)
    {
        //remove softmax
        if (graph == null){
            ((BaseLayer)network.getLayer("output").getConfig()).setActivationFn(new ActivationIdentity());
        }else{
            ((BaseLayer)graph.getLayer("output").getConfig()).setActivationFn(new ActivationIdentity());
        }
        
        //get original stride and padding
        int stride = 1; // TODO: 12/30/2020
        int padding = 1;

        //generate sized random image
        INDArray input = Nd4j.rand(1, chann, height, width);

        //its already normed so we reverse if desired
        if (!norm) input.muli(255);

        //hold params
        Map<String, INDArray> params;
        if (graph == null){
         params = network.paramTable();
        }else{
         params = graph.paramTable();
        }


        // calc against output data
        if (graph == null){
             network.fit(input, out);

        }else{
            graph.fit(new INDArray[] {input}, new INDArray[] {out});
        }


        //calc gradient
        Gradient error  = null;
        if (graph == null){
            error = network.gradient();
                    network.computeGradientAndScore();
        }else{
            error = graph.gradient();
        }


        //adjust image
        INDArray layer0 = error.getGradientFor("0_W"); //input layer
        layer0.muli(-1); //ascent
        layer0.mean(0);

        // TODO: 12/30/2020 this needs o perform an elementwise sum as it strides across
        //


        //reset oarams
        if (graph == null){
            network.setParamTable(params);
        }else{
            graph.setParamTable(params);
        }



    }


    //load the kernel visuals
    public void update() {
        boolean buildUpList = false;
        if (upsampleList.size() == 0) {
            buildUpList = true;
        }

        if (network == null){
            paramVar = graph.paramTable();
        }else{
            paramVar = network.paramTable();
        }


        if (paramVar == null) return;


        for (int i = 0; i < layersToView.size(); i++) {

            //get layer
            String val = "";
            if (network == null){
                val = namesList.get(layersToView.get(i)) + "_W";
            }else{
                val = layersToView.get(i) + "_W";
            }

            INDArray first = paramVar.get(val);
            if (first == null){
                System.out.println("Null kernels at " + val + "  " + i);
                upsampleList.add(null);
                continue;
            }

            //compress eachlayer into 2d activation map
            first = first.sum(1);

            //norm between zero and 1
            Number min = first.minNumber();
            Number max = first.maxNumber();
            first = first.sub(min).div((double) max - (double) min);


            //all layouts are 8 x 8
            first = correctWeirdArrays(first, val);
            int rows = (int) first.size(0) / 8;
            first = first.reshape(1, 1, first.size(1) * rows, first.size(2) * 8);


            // TODO: 12/21/2020 After the first layer this needs to build lower layer features into higher layer
            // bring the activations into visible 255 - 0 range
            INDArray colorKernel = rgbKernel((int) first.size(0),
                    (int) first.size(1), (int) first.size(2), (int) first.size(3));
            first.muli(colorKernel);

            //drop 4th dimension so image lader can read it
            //first = first.get(NDArrayIndex.interval(0,1), NDArrayIndex.all(),
            //NDArrayIndex.all(), NDArrayIndex.all());

            if (buildUpList) {

                upsampleList.add(ResizeNets.buildEnlargeNet(4, (int) first.size(0),
                        (int) first.size(1) * rows, (int) first.size(2) * 8));

                //INDArray d = null;
                //outputList.add(d);
            }

            //outputList.set(i, upsampleList.get(i).outputSingle(first));

            BufferedImage bf = loadersList.get(i).asBufferedImage(upsampleList.get(i).outputSingle(first));

            labelsList.get(i).setIcon(new ImageIcon(bf));

        }
    }



    private void builPanel()
    {
        int screen = 2;

        GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
        GraphicsDevice[] gd = ge.getScreenDevices();
        GraphicsDevice graphicsDevice;
        if( screen > -1 && screen < gd.length ) {
            graphicsDevice = gd[screen];
        } else if( gd.length > 0 ) {
            graphicsDevice = gd[0];
        } else {
            throw new RuntimeException( "No Screens Found" );
        }
        Rectangle bounds = graphicsDevice.getDefaultConfiguration().getBounds();
        int screenWidth = graphicsDevice.getDisplayMode().getWidth();
        int screenHeight = graphicsDevice.getDisplayMode().getHeight();






        tabbedPane.setBounds(0,0,maxDimensions, maxDimensions);

        for (int i = 0; i < layersToView.size(); i++) {

            JPanel p = new JPanel();
            panelList.add(p);
            JLabel l = new JLabel();
            labelsList.add(l);

            Java2DNativeImageLoader IM = new Java2DNativeImageLoader();
            loadersList.add(IM);

            p.add(l);
            tabbedPane.add(new JScrollPane(p), namesList.get(layersToView.get(i)));
            //tabbedPane.add(namesList.get(layersToView.get(i)), p);


        }


        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.add(tabbedPane);
        f.setSize(maxDimensions + 16 , maxDimensions + 16);
        f.setEnabled(true);
        f.setLayout(null);
        f.setLocation(bounds.x + (screenWidth - f.getPreferredSize().width) / 2,
                bounds.y + (screenHeight - f.getPreferredSize().height) / 2);

        f.setVisible(true);


    }



    //activation
    private void updateResults(INDArray workArr, int i)
    {


        //norm between zero and 1
        Number min =  workArr.minNumber();
        Number max = workArr.maxNumber();
        workArr = workArr.sub(min).div ( (double) max - (double) min);


        //all layouts are 8 x 8
       workArr = correctWeirdArrays(workArr, String.valueOf(i));

        int rows = (int) workArr.size(1) / 8;
        workArr = workArr.reshape(1,1,workArr.size(2) * rows, workArr.size(3) * 8);



        // bring the activations into visible 255 - 0 range
        INDArray colorKernel = rgbKernel((int) workArr.size(0),
                (int)workArr.size(1) , (int)workArr.size(2), (int)workArr.size(3));
        workArr.muli(colorKernel);

        if (upsamples.size() < i + 1){

            upsamples.add(ResizeNets.buildEnlargeNet(2, (int)workArr.size(1) ,
                    (int)workArr.size(2), (int)workArr.size(3)));

        }

        workArr = upsamples.get(i).outputSingle(workArr);

        BufferedImage bf = loadersList.get(i).asBufferedImage(workArr);

        labelsList.get(i).setIcon(new ImageIcon(bf));
    }


    // a tensor with all vals at 255 B,G,R
    private INDArray rgbKernel(int batch,int depth, int height, int width)
    {
        INDArray basic = Nd4j.ones(batch,  depth,height, width);

        basic.muli(255);

        return basic;

    }


    private INDArray correctWeirdArrays(INDArray workArr, String ident)
    {
        if (workArr.shape().length == 4) {

            while ((int) workArr.size(1) % 8 != 0) {
                workArr = workArr.get(NDArrayIndex.all(),
                        NDArrayIndex.interval(0, workArr.size(1) - 1),
                        NDArrayIndex.all(), NDArrayIndex.all());
                System.out.println("Filter removed for fitting " + ident);
            }

        }else
        {

            while ((int) workArr.size(0) % 8 != 0) {
                workArr = workArr.get(NDArrayIndex.interval(0, workArr.size(0) - 1),
                        NDArrayIndex.all(), NDArrayIndex.all());
                System.out.println("Filter removed for fitting " + ident);
            }

        }

        return workArr;

    }

    //endregion


}
