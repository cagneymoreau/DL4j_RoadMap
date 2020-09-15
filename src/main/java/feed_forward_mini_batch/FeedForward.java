package feed_forward_mini_batch;

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class FeedForward {

    MultiLayerNetwork network;

    public FeedForward(){

    }





    //region-----------------------------simple regression
    public DataSet ttLinearRegression(DataSet dataSet){

        buildNet();

        dataSet.setFeatures(dataSet.getFeatures().reshape(dataSet.getFeatures().length(), 1));
        dataSet.setLabels(dataSet.getLabels().reshape(dataSet.getLabels().length(),1));
        dataSet.dataSetBatches(100);

        DataSetIterator iter = new ExistingDataSetIterator(dataSet);
        network.fit(iter, 10);


        NormalizerMinMaxScaler n = new NormalizerMinMaxScaler();
        n.fit(dataSet);

        INDArray x = Nd4j.linspace(Math.round(n.getMin().getFloat(0)), Math.round(n.getMax().getFloat(0)), 10).reshape(10, 1);
        INDArray y = network.output(x);

        DataSet result = new DataSet();
        result.setFeatures(x);
        result.setLabels(y);

        return result;

    }


    private void buildNet()
    {
        int seed = 12345;
        int numInputs = 1;
        int numOutputs = 1;


        // Hook up one input to the one output.
        // The resulting model is a straight line.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(.003, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numOutputs)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.L1)
                        .activation(Activation.IDENTITY)
                        .nIn(numOutputs).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));


        network = net;


    }

    //endregion


    //region----------------------------regressionMathfunction

    public DataSet ttFuncRegression(DataSet dataSet){

        diffNet();

        dataSet.setFeatures(dataSet.getFeatures().reshape(dataSet.getFeatures().length(), 1));
        dataSet.setLabels(dataSet.getLabels().reshape(dataSet.getLabels().length(),1));
        //dataSet.dataSetBatches(50);
        final List<DataSet> list = dataSet.asList();
        Collections.shuffle(list, new Random(123));

        DataSetIterator iter = new ListDataSetIterator<>(list, 50);
        //DataSetIterator iter = new ExistingDataSetIterator(dataSet);
        network.fit(iter, 500);


        NormalizerMinMaxScaler n = new NormalizerMinMaxScaler();
        n.fit(dataSet);

        int number = 20;
        INDArray x = Nd4j.linspace(Math.round(n.getMin().getFloat(0)), Math.round(n.getMax().getFloat(0)), number).reshape(number, 1);
        INDArray y = network.output(x);

        DataSet result = new DataSet();
        result.setFeatures(x);
        result.setLabels(y);

        return result;

    }


    private void diffNet(){

        int seed = 12345;
        int numInputs = 1;
        int numOutputs = 1;
        int numHidden = 50;



        // Hook up one input to the one output.
        // The resulting model is a straight line.

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(.01, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHidden).nOut(numHidden)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.L2)
                        .activation(Activation.IDENTITY)
                        .nIn(numHidden).nOut(numOutputs).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(5));


        network = net;


    }

    //endregion


    //region---------------------XOR

    public void ttXor(DataSet ds)
    {
        xorNet();

        DataSetIterator di = new ExistingDataSetIterator(ds);

        network.fit(di, 2000);

        INDArray output = network.output(ds.getFeatures());
        System.out.println(output);

    }

    private void xorNet(){

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.5))
                .seed(1234)
                .biasInit(0) // init the bias with 0 - empirical value, too
                // from "http://deeplearning4j.org/architecture": The networks can
                // process the input more quickly and more accurately by ingesting
                // minibatches 5-10 elements at a time in parallel.
                // this example runs better without, because the dataset is smaller than
                // the mini batch size
                .miniBatch(false)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(4)
                        .activation(Activation.SIGMOID)
                        // random initialize weights with values between 0 and 1
                        .weightInit(new UniformDistribution(0, 1))
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .weightInit(new UniformDistribution(0, 1))
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // add an listener which outputs the error every 100 parameter updates
        net.setListeners(new ScoreIterationListener(100));

        // C&P from LSTMCharModellingExample
        // Print the number of parameters in the network (and for each layer)
        System.out.println(net.summary());

        network = net;


    }

    //endregion





}
