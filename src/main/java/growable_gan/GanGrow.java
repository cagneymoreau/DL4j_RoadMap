package growable_gan;

import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Triple;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 *
 * This is my first attempt. It runs but doesnt work
 *
 */


public class GanGrow {


    double learnRate = 1e-3;
    int height = 32;
    int width = height;
    int latentDim = 10;
    String save = "D:\\Downloads\\img_align_celeba\\ganOut\\";
    String source = "D:\\Downloads\\img_align_celeba\\img_align_celeba\\";

    DataSetIterator picIter;
    GaussianIterator gaussianIterator;

    INDArray[] labelTrue;
    INDArray[] labelFalse;

    ComputationGraph discriminatorNet;
    ComputationGraph generatorNet;
    ComputationGraph fullGanNet;
    ComputationGraph shrinkNet;

    Java2DNativeImageLoader imageLoader;

    int[] fSwitch = new int[3];

    public GanGrow()
    {
        imageLoader = new Java2DNativeImageLoader();
        labelFalse = new INDArray[] {Nd4j.zeros(1,1)};
        labelTrue = new INDArray[] {Nd4j.ones(1, 1)};
    }


    public void run()
    {
        if (width % 4 != 0){
            System.err.println("Dimensions must be multiple of 4");
            return;
        }

        picIter = getIter();
        gaussianIterator = new GaussianIterator(latentDim, latentDim);

        int grow = calcGrowLevels(); //how many growth cycles ex 32 = 3 cycles + final

        for (int i = 0; i <= grow; i++) {

            //build layers and train this level
                buildNetworks(i, grow);
                trainNetworks();
        }


    }

    //region training

    private void trainNetworks()
    {
        boolean unkownBenchmark = true;
        int epoch = 0;
        while (unkownBenchmark){

            trainDisc(epoch);
            trainGan();

            epoch++;
        }

    }

    private void trainDisc(int epoch)
    {
        System.out.println("Train Disc");

        boolean keepTrain = true;
        int loopcount = 0;  //if we get stick in a loop we can generate a warning to user

        while (keepTrain){

            if (!picIter.hasNext()){
                picIter.reset();
            }

            // true sample
            INDArray fullImg = picIter.next().getFeatures(); //get full pic
            INDArray[] subImg = shrinkNet.output(fullImg);  //get scaled trainer
            Map<String, INDArray> outG = discriminatorNet.feedForward (subImg, 0, false);
            INDArray p = discriminatorNet.params();
            INDArray[] resTruth = discriminatorNet.output(subImg);
            discriminatorNet.fit(subImg, labelTrue);
            Gradient g = discriminatorNet.gradient();


            //fake sample
            Map<String, INDArray> outB = generatorNet.feedForward(gaussianIterator.next(),1, false);
            INDArray[] fakeSub = generatorNet.output(gaussianIterator.next());
            INDArray[] resFake = discriminatorNet.output(fakeSub);
            discriminatorNet.fit(fakeSub, labelFalse);

            int result = 0;
            if (resTruth[0].getFloat(0) - resFake[0].getFloat(0) > .5f){
                result = 1;
            }

            keepTrain = lastThree(result);

            loopcount++;
            if (loopcount > 100){
                System.out.println("Stuck Training Disc");
                loopcount = 0;
            }

            //save
            if (!keepTrain){

                INDArray tosave = Nd4j.concat(3,subImg[0], fakeSub[0]);
                saveImage(tosave, "res_" + epoch);
            }

        }

    }

    private void trainGan()
    {
        System.out.println("Train Gen");

        //load our discriminator params into frozen network
        org.deeplearning4j.nn.api.Layer[] lCnt = generatorNet.getLayers();
        org.deeplearning4j.nn.api.Layer[] layers = discriminatorNet.getLayers();
        for (int i = lCnt.length; i < (layers.length + lCnt.length); i++) {

            fullGanNet.getLayer(i).setParams(layers[i-lCnt.length].params());

        }



        //perform updates of fullnetwork
        boolean keepTrain = true;
        int loopcount = 0;

        while(keepTrain){

            Map<String, INDArray> outG = fullGanNet.feedForward (gaussianIterator.next(), 2, false);
            INDArray[] res = fullGanNet.output(gaussianIterator.next());
            fullGanNet.fit(gaussianIterator.next(), labelTrue);
            Gradient g = fullGanNet.gradient();

            int result = 0;
            if (res[0].getFloat(0) > .5f){
                result = 1;
            }
            keepTrain = lastThree(result);

            loopcount++;
            if (loopcount > 100){
                System.out.println("Stuck Training Gen");
            }

        }
        //update the gan so that it has its new learned strategy
        for (int i = 0; i < lCnt.length; i++) {
            generatorNet.getLayer(i).setParams(fullGanNet.getLayer(i).params());
        }

    }



    //endregion

    //region networks

    private void buildNetworks(int targetDim, int grow){

        shrinkNet = buildShrinkNet(grow - targetDim);
        generatorNet = buildGenerator(getGenLayers(), learnRate);
        discriminatorNet = buildDiscriminator(getDiscLayers(""), learnRate, 4);
        fullGanNet = buildGan(getGenLayers(),getDiscLayers("gen_2"), learnRate);
    }


    /**
     * Shrink picture to match gen disc input.
     * 0 shrink just pass info through without changing
     * @param shrink
     * @return
     */
    private ComputationGraph buildShrinkNet(int shrink)
    {
        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("input");
        String input = "input";
        String layerName;
        if (shrink == 0) {

            confBuild = confBuild.addLayer("sub_1", new DenseLayer.Builder().nIn(width * height).nOut(width * height)
                    .activation(Activation.IDENTITY).build(), input);

        }else {
            for (int i = 0; i < shrink; i++) {
                layerName = "sub_" + (i + 1);
                confBuild = confBuild.addLayer(layerName, new SubsamplingLayer.Builder().kernelSize(2, 2).stride(2, 2).poolingType(SubsamplingLayer.PoolingType.AVG).build(), input);
                input = layerName;
            }
        }

        confBuild
                .setOutputs(input)
                .setInputTypes(InputType.convolutional(32, 32, 3))
                .build();


        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        //net.setListeners(new ScoreIterationListener(100));
        //net.gradient()
        return net;

    }


    private List<Triple<String, Layer, String>> getGenLayers()
    {
        List<Triple<String, Layer, String>> layerList = new ArrayList<>();

        layerList.add(new Triple<>("gen_1", new ConvolutionLayer.Builder().kernelSize(5,5).stride(1,1).nIn(1).nOut(128).activation(Activation.LEAKYRELU).build(), "gen_in"));
        //28x6x6
        layerList.add(new Triple<>("gen_2", new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).nIn(128).nOut(3).activation(Activation.LEAKYRELU).build(), "gen_1"));
        //3x2x2

        return layerList;
    }




    private ComputationGraph buildGenerator(List<Triple<String, Layer, String>> layerlist, double lr)
    {
        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(lr))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("gen_in");

        for (int i = 0; i < layerlist.size(); i++) {
            confBuild = confBuild.addLayer(layerlist.get(i).getFirst(), layerlist.get(i).getSecond(), layerlist.get(i).getThird() );
        }


        confBuild
                .setOutputs("gen_2")
                //.setInputTypes(InputType.convolutional(32, 32, 3))
                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        //net.gradient()
        return net;

    }


    private List<Triple<String, Layer, String>> getDiscLayers(String combo)
    {
        List<Triple<String, Layer, String>> layerList = new ArrayList<>();

        String in = "disc_in";
        if (!combo.equals("")) in = combo;

        //layerList.add(new Triple<>("disc_1", new DenseLayer.Builder().nIn(48).nOut(48).activation(Activation.IDENTITY).build(), in));


        layerList.add(new Triple<>("output", new OutputLayer.Builder(LossFunctions.LossFunction.L2).nIn(48).nOut(1).activation(Activation.SIGMOID).build(), in));

        return layerList;

    }


    private ComputationGraph buildDiscriminator(List<Triple<String, Layer, String>> layerlist, double lr, int dim)
    {

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(lr))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs( "disc_in");

        for (int i = 0; i < layerlist.size(); i++) {
            confBuild = confBuild.addLayer(layerlist.get(i).getFirst(), layerlist.get(i).getSecond(), layerlist.get(i).getThird() );
        }


        confBuild.setOutputs("output")
                .setInputTypes(InputType.convolutional(dim, dim, 3));


        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        //net.gradient()
        return net;


    }


    private ComputationGraph buildGan(List<Triple<String, Layer, String>> genlist, List<Triple<String, Layer, String>> discList, double lr)
    {

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(lr))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("gen_in");

        for (int i = 0; i < genlist.size(); i++) {
            confBuild = confBuild.addLayer(genlist.get(i).getFirst(), genlist.get(i).getSecond(), genlist.get(i).getThird() );
        }

        for (int i = 0; i < discList.size(); i++) {
            confBuild = confBuild.addLayer(discList.get(i).getFirst(), new FrozenLayerWithBackprop(discList.get(i).getSecond()), discList.get(i).getThird() );
        }
        //confBuild  = confBuild.addLayer(discList.get(discList.size()-1).getFirst(), discList.get(discList.size()-1).getSecond(), discList.get(discList.size()-1).getThird() );



        confBuild.setOutputs("output")
                //.setInputTypes(InputType.convolutional(32, 32, 3))
                .inputPreProcessor("output", new CnnToFeedForwardPreProcessor(4,4, 3));


        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        //net.gradient()

        return net;


    }

    //endregion



    //region -------- iters/helpers

    private boolean lastThree(int v)
    {
        fSwitch[0] = fSwitch[1];
        fSwitch[1] = fSwitch[2];
        fSwitch[2] = v;

        int ready = 0;
        for (int i = 0; i < fSwitch.length; i++) {
            ready += fSwitch[i];
        }
        if (ready == 3){
            fSwitch[2] = 0;
            return false;
        }

        return true;
    }


    //testing
    public void reducePicture()
    {
        DataSetIterator picIter = getIter();

        ComputationGraph net = buildShrinkNet(1);

        INDArray data = picIter.next().getFeatures();

        INDArray[] out = net.output(data);

        INDArray iii = new GaussianIterator(10,10).next()[0];

        saveImage(iii, "truth");
        //aveImage(out[0], "sample");
        System.out.println(" - ");

    }




    private int calcGrowLevels()
    {
        int target = width;
        int sizeLevels = 0;

        while (target > 4)
        {
            target = target/2;
            sizeLevels++;
        }

        return sizeLevels;
    }


    public DataSetIterator getIter()
    {
        try{
            ImageRecordReader recordReader = new ImageRecordReader(height,width, 3);
            recordReader.initialize(new FileSplit(new File(source)));

            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 1);

            return dataSetIterator;

        }catch (Exception e){
            e.printStackTrace();
        }

        return null;


    }



    private void saveImage(INDArray im, String name)
    {
        BufferedImage bf = imageLoader.asBufferedImage(im);


        File f = new File(save + name + ".jpg");

        try {
            ImageIO.write(bf, "jpg", f);
        }catch (Exception e)
        {
            System.out.print(" ");
        }


    }



    private class GaussianIterator implements Iterator<INDArray[]> {


        int width;
        int height;

        public GaussianIterator(int w, int h){
            width = w;
            height = h;
        }

        public void setWidth(int w){
            width = w;
        }

        public void setHeight(int h)
        {
            height = h;
        }


        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public INDArray[] next() {
            return new INDArray[] {getArr()};
        }


        private INDArray getArr(){

            INDArray ind = Nd4j.rand(128,1,1);
            ind = ind.mul(255);
            return ind;

        }



    }


        //endregion


}
