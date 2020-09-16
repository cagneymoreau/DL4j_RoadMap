package growable_gan;

import org.apache.commons.io.FileUtils;
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
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Triple;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 *
 * Made this to try different thing
 * I think need to to do an overfit to prove there is not a bug
 * but I think theres a bug
 *
 *
 *
 * dropout .3 on disc?
 */

public class FastGantwo {

    //region ----------------  vars



    private static JFrame frame;
    JLabel originalLaabel;
    JLabel resultLabel;
    Java2DNativeImageLoader imageLoader;
    Java2DNativeImageLoader imageLoaderTwo;


    double dropout = 0.3;
    double learnRate = 1e-3;
    int height = 32;
    int width = height;
    int latentDim = 10;
    String save = "D:\\Downloads\\img_align_celeba\\ganOut\\";
    String source = "D:\\Downloads\\img_align_celeba\\img_align_celeba\\";
    int maxcycles = 1024; //64;

    DataSetIterator picIter;
    GaussianIterator gaussianIterator;

    INDArray[] labelTrue;
    INDArray[] labelFalse;

    ComputationGraph discriminatorNet;
    ComputationGraph generatorNet;
    ComputationGraph fullGanNet;
    ComputationGraph shrinkNet;

    int minibatch = 1;

    int[] fSwitch = new int[3];

    //In order to debug lets try overfitting on a single image
    INDArray[] subImg; //here is our ground truth
    INDArray[] fakeSub; //here is our randoms


    //endregion


    //region -------------------- run

    public FastGantwo()
    {
        imageLoader = new Java2DNativeImageLoader();
        labelFalse = new INDArray[] {Nd4j.zeros(minibatch,1)};
        labelTrue = new INDArray[] {Nd4j.ones(minibatch, 1)};

        deleteFolder();

        buildWindow();

    }


    public void run()
    {

        //List<Triple<String, Layer, String>> h = getGenLayers(1);

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
            trainNetworks(i);
        }


    }

    public void test()
    {

        picIter = getIter();
        shrinkNet = buildShrinkNet(3);
        generatorNet = buildGenerator(getGenLayers(0, true), learnRate);
        INDArray[] pic = new INDArray[]  {picIter.next().getFeatures()};
        INDArray[] fuzz = new INDArray[] {Nd4j.rand(1,512,1,1)};
        subImg = shrinkNet.output(pic);
        Map<String, INDArray> outB = generatorNet.feedForward(fuzz,generatorNet.getLayers().length-1, false);


        for (int i = 0; i < 20000; i++) {
            generatorNet.fit(fuzz, subImg);
            INDArray[] out = generatorNet.output(fuzz);
            displayImage(out[0],0,"blank");

        }



    }

    //endregion


    //region -------------------- training

    private void trainNetworks(int level)
    {

        boolean unkownBenchmark = true; //disc cant tell difference
        int epoch = 0;
        while (unkownBenchmark) {

            unkownBenchmark = trainDisc(epoch, level);
            trainGan(level);

            epoch++;
        }
        System.out.println("Leveled up at epoch: " + epoch);
        maxcycles = maxcycles * 3;
    }

    private boolean trainDisc(int epoch, int level)
    {
        System.out.println("Train Disc");

        boolean keepTrain = true;
        int loopcount = 0;  //if we get stuck in a loop we can generate a warning to user

        float truthAprox = 0;
        float fakeAprox = 0;

        while (keepTrain){

            if (!picIter.hasNext()){
                picIter.reset();
            }

            // true sample
            // TODO: 8/18/2020 only train it to overfit a single image

            Map<String, INDArray> outG = discriminatorNet.feedForward (subImg, discriminatorNet.getLayers().length-2, false);
            //INDArray p = discriminatorNet.params();
            INDArray[] resTruth = discriminatorNet.output(subImg);
            discriminatorNet.fit(subImg, labelTrue);
            //Gradient g = discriminatorNet.gradient();


            //fake sample
            Map<String, INDArray> outB = generatorNet.feedForward(gaussianIterator.next(),generatorNet.getLayers().length-1, false);
            INDArray[] fake = generatorNet.output(fakeSub); //show what we have overfit too
            INDArray[] resFake = discriminatorNet.output(Nd4j.rand(minibatch,3, 4,4));
            discriminatorNet.fit(new INDArray[] {Nd4j.rand(minibatch, 3,4,4)}, labelFalse);

            int result = 0;
            if (resTruth[0].getFloat(0) - resFake[0].getFloat(0) > .8f){
                result = 1;
            }

            keepTrain = lastThree(result);

            loopcount++;
            if (loopcount > maxcycles){
                System.out.println("Stuck Training Disc");
                return false;
                //loopcount = 0;

            }

            INDArray ex = subImg[0].get(NDArrayIndex.point(0), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all());
            INDArray ex2 = fake[0].get(NDArrayIndex.point(0), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all());

            //truthAprox += resTruth[0].getFloat(0);
            //fakeAprox +=  resFake[0].getFloat(0);

            String desc = resTruth[0].getFloat(0) + "  " + resFake[0].getFloat(0);
            //save
            INDArray tosave = Nd4j.concat(2,ex, ex2);

            displayImage(tosave, level, desc);

            if (!keepTrain){
                saveImage(tosave, "res_" + level + "_" + epoch);
            }

        }

        return true;

    }

    private void trainGan(int level)
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

            Map<String, INDArray> outG = fullGanNet.feedForward (gaussianIterator.next(), fullGanNet.getLayers().length-1, false);
            INDArray[] res = fullGanNet.output(fakeSub);
            fullGanNet.fit(fakeSub, labelTrue);
            //Gradient g = fullGanNet.gradient();

            INDArray ex = subImg[0].get(NDArrayIndex.point(0), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all());
            INDArray ex2 = outG.get("toRGB").get(NDArrayIndex.point(0), NDArrayIndex.all(),NDArrayIndex.all(), NDArrayIndex.all());
            String desc = String.valueOf(res[0].getFloat(0));
            INDArray tosave = Nd4j.concat(2,ex, ex2);
            displayImage(tosave, level, desc);

            int result = 0;
            if (res[0].getFloat(0) > .9f){
                result = 1;
            }
            keepTrain = lastThree(result);

            loopcount++;
            if (loopcount > maxcycles){
                System.out.println("Stuck Training Gen");
                keepTrain = false;
                loopcount = 0;
            }

        }
        //extract the fullgan params to gen so that it has its new learned strategy
        for (int i = 0; i < lCnt.length; i++) {
            generatorNet.getLayer(i).setParams(fullGanNet.getLayer(i).params());
        }

    }



    //endregion



    //region ------------------------- networks

    private void buildNetworks(int layerCount, int grow){

        int currDim = 2;
        for (int i = 0; i <= layerCount; i++) {
            currDim = currDim * 2;
        }

        ComputationGraph genTemp = null;
        ComputationGraph disTemp = null;
        ComputationGraph ganTemp = null;

        //save old params to insert ni new network
        if (generatorNet != null){

            genTemp = generatorNet;
            disTemp = discriminatorNet;
            //ganTemp = fullGanNet;
        }

        shrinkNet = buildShrinkNet(grow - layerCount);
        generatorNet = buildGenerator(getGenLayers(layerCount, false), learnRate);
        discriminatorNet = buildDiscriminator(getDiscLayers("", layerCount, dropout), learnRate, currDim);
        fullGanNet = buildGan(getGenLayers(layerCount, false),getDiscLayers("toRGB", layerCount, 0.0), learnRate);

        //restore appropriate params
        if (genTemp != null){
            //new layer concates to end
            org.deeplearning4j.nn.api.Layer[] genlayers = genTemp.getLayers();
            for (int i = 0; i < genlayers.length; i++) {
                if (i > genlayers.length - 2){
                    generatorNet.getLayer(i+3).setParams(genlayers[i].params());
                    fullGanNet.getLayer(i+3).setParams(genlayers[i].params());
                }else{
                    generatorNet.getLayer(i).setParams(genlayers[i].params());
                    fullGanNet.getLayer(i).setParams(genlayers[i].params());
                }

            }

            org.deeplearning4j.nn.api.Layer[] disclayers = disTemp.getLayers();
            for (int i = 0; i < disclayers.length; i++) {
                if ( i > 0){
                    discriminatorNet.getLayer(i + 3).setParams(disclayers[i].params());

                }else{
                    discriminatorNet.getLayer(i).setParams(disclayers[i].params());

                }
            }





        }

        // TODO: 8/17/2020 here we create our overfit
        INDArray fullImg = picIter.next().getFeatures(); //get full pic
        subImg = shrinkNet.output(fullImg);  //get scaled trainer
        fakeSub = gaussianIterator.next();
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
            input = "sub_1";
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
        return net;

    }



    private List<Triple<String, Layer, String>> getGenLayers(int layerCount, boolean sep)
    {
        List<Triple<String, Layer, String>> layerList = new ArrayList<>();
        // TODO: 8/13/2020 could switch this to padded 4x4 per writeup
        layerList.add(new Triple<>("gen_1", new ConvolutionLayer.Builder().kernelSize(4,4).padding(3,3).stride(1,1).nIn(512).dropOut(dropout).nOut(512).activation(Activation.LEAKYRELU).build(), "gen_in"));

        layerList.add(new Triple<>("gen_2", new ConvolutionLayer.Builder() .kernelSize(3,3).padding(1,1) .stride(1,1).nIn(512).dropOut(dropout).nOut(512).activation(Activation.LEAKYRELU).build(), "gen_1"));
        //layerList.add(new Triple<>("gen_2", new ConvolutionLayer.Builder().kernelSize(3,3) .stride(1,1).nIn(512).nOut(512).activation(Activation.LEAKYRELU).build(), "gen_1"));

        String startName = "gen_2";
        String layerName = "gen_";
        int lyrCnt = 3;

        /**  Disc Layers Here **/

        for (int i = 0; i < layerCount; i++) {

            layerList.add(new Triple<>(layerName + lyrCnt, new Upsampling2D.Builder().size(2) .build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(512).dropOut(dropout).nOut(512).activation(Activation.LEAKYRELU).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(512).dropOut(dropout).nOut(512).activation(Activation.LEAKYRELU).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
        }

        /**   -   **/

        layerList.add(new Triple<>("toRGB", new ConvolutionLayer.Builder().kernelSize(1,1) .stride(1,1).nIn(512).nOut(3).activation(Activation.IDENTITY).build(), startName));
        //layerList.add(new Triple<>("toRGB", new ConvolutionLayer.Builder().kernelSize(1,1) .stride(1,1).nIn(512).nOut(3).activation(Activation.IDENTITY).build(), "gen_2"));
        if (sep) {
            layerList.add(new Triple<>("out", new CnnLossLayer.Builder().activation(Activation.IDENTITY).lossFunction(LossFunctions.LossFunction.MSE).build(), "toRGB"));
        }

        return layerList;
    }




    private ComputationGraph buildGenerator(List<Triple<String, Layer, String>> layerlist, double lr)
    {
        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(lr))
                .weightInit(WeightInit.RELU)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("gen_in");

        for (int i = 0; i < layerlist.size(); i++) {
            confBuild = confBuild.addLayer(layerlist.get(i).getFirst(), layerlist.get(i).getSecond(), layerlist.get(i).getThird() );
        }


        confBuild
                .setOutputs("out") // TODO: 8/18/2020 "toRGB"
                .build();

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        return net;

    }



    private List<Triple<String, Layer, String>> getDiscLayers(String combo, int layerCount, double drop)
    {


        List<Triple<String, Layer, String>> layerList = new ArrayList<>();

        String in = "disc_in";
        if (!combo.equals("")) in = combo;
        String startName = "fromRGB";
        String layerName = "disc_";
        int lyrCnt = 1;
        layerList.add(new Triple<>(startName, new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).nIn(3).nOut(512).activation(Activation.IDENTITY).build(), in));

        /**  Disc Layers Here **/

        for (int i = 0; i < layerCount; i++) {

            layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(512).nOut(512).dropOut(drop).activation(Activation.LEAKYRELU).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            layerList.add(new Triple<>(layerName + lyrCnt, new ConvolutionLayer.Builder().kernelSize(3,3).padding(1,1).stride(1,1).nIn(512).nOut(512).dropOut(drop).activation(Activation.LEAKYRELU).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            layerList.add(new Triple<>(layerName + lyrCnt, new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build(), startName));
            startName = layerName + lyrCnt; lyrCnt++;
            // TODO: 8/13/2020 elim padding and subsample?
        }

        /**   -   **/

        layerList.add(new Triple<>("final_1", new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).nIn(512).nOut(512).dropOut(drop).activation(Activation.LEAKYRELU).build(), startName));

        layerList.add(new Triple<>("final_2", new ConvolutionLayer.Builder().kernelSize(4,4).padding(1,1).stride(1,1).nIn(512).nOut(1).dropOut(drop).activation(Activation.LEAKYRELU).build(), "final_1"));

        layerList.add(new Triple<>("output", new OutputLayer.Builder(LossFunctions.LossFunction.L2).nIn(1).nOut(1).activation(Activation.SIGMOID).build(), "final_2"));
        // TODO: 8/12/2020 activation linear?
        return layerList;

    }



    private ComputationGraph buildDiscriminator(List<Triple<String, Layer, String>> layerlist, double lr, int dim)
    {

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(lr))
                .weightInit(WeightInit.RELU)
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



        return net;


    }



    private ComputationGraph buildGan(List<Triple<String, Layer, String>> genlist, List<Triple<String, Layer, String>> discList, double lr)
    {

        ComputationGraphConfiguration.GraphBuilder  confBuild = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new RmsProp(lr))
                .weightInit(WeightInit.RELU)
                .l2(1e-4)
                .graphBuilder()
                .addInputs("gen_in");

        for (int i = 0; i < genlist.size(); i++) {
            confBuild = confBuild.addLayer(genlist.get(i).getFirst(), genlist.get(i).getSecond(), genlist.get(i).getThird() );
        }

        for (int i = 0; i < discList.size(); i++) {
            confBuild = confBuild.addLayer(discList.get(i).getFirst(), new FrozenLayerWithBackprop(discList.get(i).getSecond()), discList.get(i).getThird() );
        }



        confBuild.setOutputs("output")
                .inputPreProcessor("output", new CnnToFeedForwardPreProcessor(1,1, 1));

        ComputationGraph net = new ComputationGraph(confBuild.build());
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        //net.gradient()

        return net;


    }


    //endregion



    //region -------- iters/helpers


    private void buildWindow()
    {

        frame = new JFrame("Results");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);


        imageLoaderTwo = new Java2DNativeImageLoader();
        originalLaabel = new JLabel();
        originalLaabel.setBounds(10,10, width * 3 , height );

        resultLabel = new JLabel();
        resultLabel.setBounds( 10,height  + 10, (width * 5) + 10, height * 2);


        frame.add(originalLaabel);
        frame.add(resultLabel);
        frame.setSize(width * 10, height * 10 );
        frame.setEnabled(true);
        frame.setLayout(null);
        frame.setVisible(true);


    }


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

    private void deleteFolder()
    {
        String[] file = new File(source).list();

        String[] files = new File(save).list();
        for (int i = 0; i < file.length; i++) {
            try {
                FileUtils.forceDelete( new File(save + files[i]));
            }catch (Exception e){

            }
        }
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

            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, minibatch);

            DataNormalization scaler = new ImagePreProcessingScaler(0,1);
            scaler.fit(dataSetIterator);
            dataSetIterator.setPreProcessor(scaler);


            return dataSetIterator;

        }catch (Exception e){
            e.printStackTrace();
        }

        return null;


    }

    public DataSetIterator getIter(int debug)
    {
        try{
            ImageRecordReader recordReader = new ImageRecordReader(height,width, 3);
            recordReader.initialize(new FileSplit(new File(source)));

            DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, minibatch);

            DataNormalization scaler = new ImagePreProcessingScaler(0,1);
            scaler.fit(dataSetIterator);
            dataSetIterator.setPreProcessor(scaler);


            return dataSetIterator;

        }catch (Exception e){
            e.printStackTrace();
        }

        return null;



    }

    private void displayImage(INDArray ind, int multiplier, String desc)
    {

        ind = ind.mul(255);

        BufferedImage before = imageLoader.asBufferedImage(ind);
        int fact = 8/(multiplier + 1);

        int w = before.getWidth() * (fact);
        int h = before.getHeight() * fact;
        BufferedImage after = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB);
        AffineTransform at = new AffineTransform();
        at.scale(fact, fact);
        AffineTransformOp scaleOp =
                new AffineTransformOp(at, AffineTransformOp.TYPE_BILINEAR);
        after = scaleOp.filter(before, after);

        originalLaabel.setIcon(new ImageIcon(after));
        resultLabel.setText(desc);
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

            INDArray ind = Nd4j.rand(minibatch,512,1,1);

            return ind;

        }



    }


    //endregion


}
