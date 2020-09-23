package dual_lstm_csv_manipulation;

import java.util.ArrayList;

/** This class will allow you to merge two csv files.
 * Each row needs a key (or multiple keys) and the imputation method is deletion
 * It places one of the files to the left of the rows and one to the right
 *
 * Use at your own risk. Not well tested
 */

public class MergeCSV {

    private int[] leftcompareColumns;
    private int[] rightcompareColumns;
    private boolean deleteComapreColumns;

    private ArrayList<Row> leftRows;
    private ArrayList<Row> rightRows;


    private StringBuilder issues = new StringBuilder();

    public MergeCSV(int[] leftcompareColumns, int[] rightcompareColumns, boolean deleteCompareColumns )
    {
        this.leftcompareColumns = leftcompareColumns;
        this.rightcompareColumns = rightcompareColumns;
        this.deleteComapreColumns = deleteCompareColumns;
        issues.append("Merge Results: \n");
    }


    public ArrayList<String> mergeFile(ArrayList<String> lRows, ArrayList<String> rRows)
    {
        leftRows = (ArrayList<Row>) buildCollection(lRows, leftcompareColumns);
        rightRows = (ArrayList<Row>) buildCollection(rRows, rightcompareColumns);

        appendData();

        return rebuildCSV();


    }

    //build row objects
    private ArrayList<Row> buildCollection(ArrayList<String> rows, int[] keyPos)
    {
        ArrayList<Row> rowList = new ArrayList<>();

        for (int i = 0; i < rows.size(); i++) {

            String[] parts = rows.get(i).split(",");
            String[] key = new String[keyPos.length];

            for (int j = 0; j < key.length; j++) {
                key[j] = parts[keyPos[j]];
            }

            String[] body = new String[parts.length- keyPos.length];

            int count = 0;
            for (int j = 0; j < parts.length; j++) {
                boolean add = true;
                for (int k = 0; k < keyPos.length; k++) {
                    if (keyPos[k] == j) add = false;
                }

                if (add){
                    body[count] = parts[j];
                    count++;
                }
            }

            Row r = new Row(key, body);
            rowList.add(r);
        }

        return rowList;
    }

    //push all matching data into the left holder
    private void appendData()
    {
        for (int i = 0; i < leftRows.size(); i++) {

            for (int j = 0; j < rightRows.size(); j++) {
                leftRows.get(i).checkRightData(rightRows.get(j));
            }


        }
    }

    //write all rows that have been appended to a list of strings
    private ArrayList<String> rebuildCSV()
    {
        ArrayList<String> out = new ArrayList<>();

        for (int i = 0; i < leftRows.size(); i++) {
            if (leftRows.get(i).isSet()){
                out.add(leftRows.get(i).getDataFinal(deleteComapreColumns));
            }
        }
        //System.out.println(issues.toString());
        return out;
    }



    private class Row{

        private boolean appended;

        private String[] key;

        private String[] data;

        private String[] rightData;


        public Row(String[] key, String[] data)
        {
            this.key = key;
            this.data = data;
        }

        public void checkRightData(Row oppRow){
            if (compare(oppRow)) {

                if (rightData != null){
                    issues.append("Double ADDED ROW!");
                }
                this.rightData = oppRow.getData();
                appended = true;
            }

        }



        public boolean compare(Row row){
            boolean match = true;
            String[] oppKey = row.getKey();

            if (oppKey.length != key.length){
                match = false;
                return match;
            }

            for (int i = 0; i < key.length; i++) {
                if (!oppKey[i].equals(key[i])){
                    match = false;
                    return match;
                }

            }
            return match;
        }

        public String[] getKey()
        {
            return key;
        }

        public String[] getData()
        {
            return data;
        }

        public boolean isSet()
        {
            return appended;
        }

        public String getDataFinal(boolean deleteCompare)
        {
            StringBuilder sb = new StringBuilder();

            if (!deleteCompare){
                for (int i = 0; i < key.length; i++) {
                    sb.append(key[i]).append(",");
                }
            }

            for (int i = 0; i < data.length; i++) {
                sb.append(data[i]).append(",");
            }

            for (int i = 0; i < rightData.length; i++) {
                sb.append(rightData[i]).append(",");
            }

            sb.setLength(sb.length()-1);
            return sb.toString();
        }

    }


}
