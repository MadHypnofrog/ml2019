import com.opencsv.CSVReader;
import javafx.util.Pair;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

public class KNN {

    private static int learnAndAnswerQuery(double[][] objects, int[] classes, int queryNum, int classesNum,
                                           String distance, String kernel, String window, double windowSize) {
        BiFunction<double[], double[], Double> d = (fst, snd) -> {
            try {
                return (double)Utils.class.getDeclaredMethod(distance, double[].class, double[].class)
                        .invoke(null, fst, snd);
            } catch (Exception e) {
                return 0D;
            }
        };
        Function<Double, Double> k = x -> {
            try {
                return (double)Utils.class.getDeclaredMethod(kernel, double.class).invoke(null, x);
            } catch (Exception e) {
                return 0D;
            }
        };
        if (window.equals("fixed")) {
            return fixed(queryNum, objects, classes, windowSize, classesNum, d, k);
        } else {
            return variable(queryNum, objects, classes, windowSize, classesNum, d, k);
        }
    }

    private static int fixed(int queryNum, double[][] train, int[] values, double d, int classesNum,
                             BiFunction<double[], double[], Double> distance,
                             Function<Double, Double> kernel) {
        List<Pair<Double, Integer>> distances = new ArrayList<>();
        double[] query = train[queryNum];
        for (int i = 0; i < train.length; i++) {
            if (i != queryNum) {
                distances.add(new Pair<>(distance.apply(query, train[i]), i));
            }
        }
        double[] scores = new double[classesNum];
        for (Pair<Double, Integer> p: distances) {
            double weight = kernel.apply(p.getKey() / d);
            scores[values[p.getValue()] - 1] += weight;
        }
        double max = 0;
        int result = 0;
        for (int i = 0; i < classesNum; i++) {
            if (scores[i] > max) {
                max = scores[i];
                result = i + 1;
            }
        }
        return result;
    }

    private static int variable(int queryNum, double[][] train, int[] values, double k, int classesNum,
                                BiFunction<double[], double[], Double> distance,
                                Function<Double, Double> kernel) {
        List<Pair<Double, Integer>> distances = new ArrayList<>();
        double[] query = train[queryNum];
        for (int i = 0; i < train.length; i++) {
            if (i != queryNum) {
                distances.add(new Pair<>(distance.apply(query, train[i]), i));
            }
        }
        distances.sort(Comparator.comparing(Pair::getKey));
        double d = distances.get((int)k).getKey();
        double[] scores = new double[classesNum];
        for (Pair<Double, Integer> p: distances) {
            double weight = kernel.apply(p.getKey() / d);
            scores[values[p.getValue()] - 1] += weight;
        }
        double max = 0;
        int result = 0;
        for (int i = 0; i < classesNum; i++) {
            if (scores[i] > max) {
                max = scores[i];
                result = i + 1;
            }
        }
        return result;
    }

    private static double calculateFMeasureForSetup(double[][] objects, int[] classes, int classesNum,
                                                    String distance, String kernel, String window, double windowSize) {
        int[][] cm = new int[classesNum][classesNum];
        for (int i = 0; i < objects.length; i++) {
            int res = learnAndAnswerQuery(objects, classes, i, classesNum, distance, kernel, window, windowSize);
            if (res == 0) return -1;
            cm[res - 1][classes[i] - 1]++;
        }
        return Utils.fmeasure(cm);
    }

    public static void main(String[] args) throws IOException {
        BufferedWriter log = new BufferedWriter(new FileWriter("log.txt"));
        List<String[]> values = new CSVReader(new FileReader("src/main/resources/dataset_16_mfeat-karhunen.csv")).readAll();
        values.remove(0);  // header
        int samples = values.size();
        int attributes = values.get(0).length - 1;
        double[][] objects = new double[samples][attributes];
        int[] classes = new int[samples];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < attributes; j++) {
                objects[i][j] = Double.valueOf(values.get(i)[j]);
            }
            classes[i] = Integer.valueOf(values.get(i)[attributes]);
        }
        int classesNum = (int)Arrays.stream(classes).distinct().count();
        for (int i = 0; i < attributes; i++) {
            double max = 0;
            for (int j = 0; j < samples; j++) {
                if (max < Math.abs(objects[j][i])) max = Math.abs(objects[j][i]);
            }
            for (int j = 0; j < samples; j++) {
                objects[j][i] /= max;
            }
        }
        String[] distances = new String[] {
                  "euclidean"
                , "manhattan"
                , "chebyshev"
        };
        String[] kernels = new String[] {
                  "uniform"
                , "triangular"
                , "epanechnikov"
                , "quartic"
                , "triweight"
                , "tricube"
                , "gaussian"
                , "cosine"
                , "logistic"
                , "sigmoid"
        };
        double max = 0;
        String maxSetup = "";
        for (String distance: distances) {
            for (String kernel: kernels) {
                XYSeriesCollection ds = new XYSeriesCollection();
                XYSeries series = new XYSeries("");
                double lower = 0;
                double upper = 0;
                double step = 0;
                int neighbors = 0;
                switch (distance) {
                    case "euclidean": {
                        lower = 0.02;
                        upper = 3.5;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) upper = 1;
                        step = 0.02;
                        neighbors = 60;
                        break;
                    }
                    case "manhattan": {
                        lower = 15;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) lower = 0.05;
                        upper = 20;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) upper = 6;
                        step = 0.05;
                        neighbors = 80;
                        break;
                    }
                    case "chebyshev": {
                        lower = 0.01;
                        upper = 1.5;
                        if (kernel.equals("gaussian") || kernel.equals("logistic") || kernel.equals("sigmoid")) upper = 1;
                        step = 0.01;
                        neighbors = 60;
                        break;
                    }
                }
                for (double window = lower; window < upper; window += step) {
                    double f = calculateFMeasureForSetup(objects, classes, classesNum, distance, kernel, "fixed", window);
                    if (f == -1.0) continue;
                    if (f < 0.75) {
                        log.write("For " + distance + " distance, " + kernel + " kernel, fixed window of "
                                + String.format("%.4f", window) + ": F1 measure is " + String.format("%.6f", f) + "\n");
                        break;
                    }
                    if (f > max) {
                        max = f;
                        maxSetup = distance + "-" + kernel + "-fixed-" + window;
                    }
                    series.add(window, f);
                    log.write("For " + distance + " distance, " + kernel + " kernel, fixed window of "
                            + String.format("%.4f", window) + ": F1 measure is " + String.format("%.6f", f) + "\n");
                }
                log.flush();
                ds.addSeries(series);
                Utils.writeChartForDS(distance + "-" + kernel + "-fixed", ds, "Window size", "F");

                XYSeriesCollection ds1 = new XYSeriesCollection();
                XYSeries series1 = new XYSeries("");
                for (int k = 1; k < neighbors; k++) {
                    double f = calculateFMeasureForSetup(objects, classes, classesNum, distance, kernel, "variable", k);
                    if (f == -1.0) continue;
                    if (f < 0.75) {
                        log.write("For " + distance + " distance, " + kernel + " kernel, variable window of " + k
                                + " neighbors: F1 measure is " + String.format("%.6f", f) + "\n");
                        break;
                    }
                    if (f > max) {
                        max = f;
                        maxSetup = distance + "-" + kernel + "-variable-" + k;
                    }
                    series1.add(k, f);
                    log.write("For " + distance + " distance, " + kernel + " kernel, variable window of " + k
                            + " neighbors: F1 measure is " + String.format("%.6f", f) + "\n");
                }
                log.flush();
                ds1.addSeries(series1);
                Utils.writeChartForDS(distance + "-" + kernel + "-variable", ds1, "Window size", "F");
            }
        }
        log.write("Optimal combination: " + maxSetup + ": " + max + "\n");
        log.flush();
    }

}
