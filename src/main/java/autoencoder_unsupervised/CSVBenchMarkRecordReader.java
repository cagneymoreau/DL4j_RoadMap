package autoencoder_unsupervised;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.metadata.RecordMetaDataLine;
import org.datavec.api.records.reader.impl.LineRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.SerializableCSVParser;
import org.datavec.api.split.InputSplit;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;

import org.nd4j.shade.protobuf.common.base.Preconditions;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

public class CSVBenchMarkRecordReader extends LineRecordReader {
    private boolean skippedLines;
    protected int skipNumLines;
    private int skipColumns;
    public static final char DEFAULT_DELIMITER = ',';
    public static final char DEFAULT_QUOTE = '"';
    public static final String SKIP_NUM_LINES;
    public static final String DELIMITER;
    public static final String QUOTE;
    private SerializableCSVParser csvParser;

    public CSVBenchMarkRecordReader(int skipNumLines, int skipColumns) {
        this(skipNumLines, ',', skipColumns);
    }

    public CSVBenchMarkRecordReader(char delimiter) {
        this(0, delimiter, 0);
    }

    public CSVBenchMarkRecordReader(int skipNumLines, char delimiter, int skipColumns) {
        this(skipNumLines, delimiter, '"', skipColumns);
    }

    /** @deprecated */
    @Deprecated
    public CSVBenchMarkRecordReader(int skipNumLines, String delimiter) {
        this(skipNumLines, stringDelimToChar(delimiter));
    }

    private static char stringDelimToChar(String delimiter) {
        if (delimiter.length() > 1) {
            throw new UnsupportedOperationException("Multi-character delimiters have been deprecated. For quotes, use CSVRecordReader(int skipNumLines, char delimiter, char quote)");
        } else {
            return delimiter.charAt(0);
        }
    }

    public CSVBenchMarkRecordReader(int skipNumLines, char delimiter, char quote, int skipColumns) {
        this.skippedLines = false;
        this.skipNumLines = 0;
        this.skipColumns = skipColumns;
        this.skipNumLines = skipNumLines;
        this.csvParser = new SerializableCSVParser(delimiter, quote);
    }

    /** @deprecated */
    @Deprecated
    public CSVBenchMarkRecordReader(int skipNumLines, String delimiter, String quote) {
        this(skipNumLines, stringDelimToChar(delimiter), stringDelimToChar(quote));
    }

    public CSVBenchMarkRecordReader() {
        this(0, ',');
    }

    public void initialize(Configuration conf, InputSplit split) throws IOException, InterruptedException {
        super.initialize(conf, split);
        this.skipNumLines = conf.getInt(SKIP_NUM_LINES, this.skipNumLines);
        this.csvParser = new SerializableCSVParser(conf.getChar(DELIMITER, ','), conf.getChar(QUOTE, '"'));
    }

    private boolean skipLines() {
        if (!this.skippedLines && this.skipNumLines > 0) {
            for(int i = 0; i < this.skipNumLines; ++i) {
                if (!super.hasNext()) {
                    return false;
                }

                super.next();
            }

            this.skippedLines = true;
        }

        return true;
    }

    public boolean batchesSupported() {
        return true;
    }

    public boolean hasNext() {
        return this.skipLines() && super.hasNext();
    }

    public List<List<Writable>> next(int num) {
        List<List<Writable>> ret = new ArrayList(Math.min(num, 10000));
        int var3 = 0;

        while(this.hasNext() && var3++ < num) {
            ret.add(this.next());
        }

        return ret;
    }

    public List<Writable> next() {
        if (!this.skipLines()) {
            throw new NoSuchElementException("No next element found!");
        } else {
            String val = this.readStringLine();
            return this.parseLine(val);
        }
    }

    protected List<Writable> parseLine(String line) {
        String[] split;
        try {
            split = this.csvParser.parseLine(line);
        } catch (IOException var8) {
            throw new RuntimeException(var8);
        }

        List<Writable> ret = new ArrayList();

        String[] var4 = split;
        int var5 = split.length;

        for(int var6 = skipColumns; var6 < var5; ++var6) {
            String s = var4[var6];
            ret.add(new Text(s));
        }

        return ret;
    }

    protected String readStringLine() {
        Preconditions.checkState(this.initialized, "RecordReader has not been initialized before use");
        Text t = (Text)super.next().iterator().next();
        return t.toString();
    }

    public Record nextRecord() {
        List<Writable> next = this.next();
        URI uri = this.locations != null && this.locations.length >= 1 ? this.locations[this.splitIndex] : null;
        RecordMetaData meta = new RecordMetaDataLine(this.lineIndex - 1, uri, CSVRecordReader.class);
        return new org.datavec.api.records.impl.Record(next, meta);
    }

    public Record loadFromMetaData(RecordMetaData recordMetaData) throws IOException {
        return (Record)this.loadFromMetaData(Collections.singletonList(recordMetaData)).get(0);
    }

    public List<Record> loadFromMetaData(List<RecordMetaData> recordMetaDatas) throws IOException {
        List<Record> list = super.loadFromMetaData(recordMetaDatas);
        Iterator var3 = list.iterator();

        while(var3.hasNext()) {
            Record r = (Record)var3.next();
            String line = ((Writable)r.getRecord().get(0)).toString();
            r.setRecord(this.parseLine(line));
        }

        return list;
    }

    public List<Writable> record(URI uri, DataInputStream dataInputStream) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(dataInputStream));

        for(int i = 0; i < this.skipNumLines; ++i) {
            br.readLine();
        }

        String line = br.readLine();
        return this.parseLine(line);
    }

    public void reset() {
        super.reset();
        this.skippedLines = false;
    }

    protected void onLocationOpen(URI location) {
        this.skippedLines = false;
    }

    static {
        SKIP_NUM_LINES = NAME_SPACE + ".skipnumlines";
        DELIMITER = NAME_SPACE + ".delimiter";
        QUOTE = NAME_SPACE + ".quote";
    }
}