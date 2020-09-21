package recurrent_regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 * Normalization strategies
 *
 */

public class Normies {


    boolean mappingComplete = false;

    double min = 0;
    double max = 0;
    double origRange = 0;

    double adjustRange = 0;
    double edge = 0;
    double adjMin = 0;

    float factor = 1;

    public Normies()
    {

    }

    //map values between min/max
    public void linearSimple(INDArray ind){



    }

    //map values between defined range 2 = 200%
    public void linearRange(INDArray ind, double percent){

        min = ind.getDouble(0);
        max = min;

        for (int i = 1; i < ind.length(); i++) {

            if (ind.getDouble(i) < min){
                min = ind.getDouble(i);
            }

            if (ind.getDouble(i) > max){
                max = ind.getDouble(i);
            }

        }

        origRange = max - min;
        adjustRange = origRange * percent;
        edge = (adjustRange - origRange) /2; //represents each side of new lower upper bound
        adjMin = min - edge;

        factor = (float) (1/adjustRange);

        System.out.println(factor);

    }

    //each data is a percentage of change from previous data point
    public INDArray percentMap(INDArray ind){

        INDArray out = Nd4j.zeros(ind.length());
        out.putScalar(1, 0); //no change %
        double hold;

        for (int i = 1 ; i < ind.length(); i++) {

           hold = ind.getDouble(i) - ind.getDouble(i-1); //get diff
           hold = hold / ind.getDouble(i-1); //get diffs as percetage of change
            out.putScalar(i, hold);

        }

        return out;

    }


    public INDArray none(INDArray ind){

        return ind;

    }





    public INDArray convert(INDArray ind){

        INDArray ii = ind.dup();

        ii = ii.sub(adjMin); //bring each value into the adjusted range
       ii = ii.mul(factor); //multiply by factor to bring between 0 and 1

        ii = ii.reshape(1,ii.length());

        return ii;
    }

    public INDArray revert(INDArray ind){

        INDArray re = ind.dup();

      re =  re.div(factor);
       re = re.add(adjMin);

       re = re.reshape(1, re.length());

        return re;
    }




    private void checkMap()
    {
        if (mappingComplete){
            System.err.println("Attmpting to remap!");
        }
    }



}
