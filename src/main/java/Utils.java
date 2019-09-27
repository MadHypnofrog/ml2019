import javafx.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Utils {

    public static double euclidean(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) {
            res += (fst[i] - snd[i]) * (fst[i] - snd[i]);
        }
        return Math.sqrt(res);
    }

    public static double manhattan(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) {
            res += Math.abs(fst[i] - snd[i]);
        }
        return res;
    }

    public static double chebyshev(double[] fst, double[] snd) {
        double res = 0;
        for (int i = 0; i < fst.length; i++) {
            double t = Math.abs(fst[i] - snd[i]);
            if (res < t) res = t;
        }
        return res;
    }

    public static double uniform(double val) {
        if (Math.abs(val) < 1) return 1D / 2;
        else return 0;
    }

    public static double triangular(double val) {
        if (Math.abs(val) < 1) return 1D - Math.abs(val);
        else return 0;
    }

    public static double epanechnikov(double val) {
        double t = (1 - val * val);
        if (Math.abs(val) < 1) return 3 * t / 4;
        else return 0;
    }

    public static double quartic(double val) {
        double t = (1 - val * val);
        if (Math.abs(val) < 1) return 15 * t * t / 16;
        else return 0;
    }

    public static double triweight(double val) {
        double t = (1 - val * val);
        if (Math.abs(val) < 1) return 35 * t * t * t / 32;
        else return 0;
    }

    public static double tricube(double val) {
        double t = (1 - Math.abs(val) * val * val);
        if (Math.abs(val) < 1) return 70 * t * t * t / 81;
        else return 0;
    }

    public static double gaussian(double val) {
        return 1 / Math.sqrt(2 * Math.PI) * Math.exp(- val * val / 2);
    }

    public static double cosine(double val) {
        if (Math.abs(val) < 1) return Math.PI * ((long)(Math.pow(10, 9) * Math.cos(Math.PI * val / 2)) / Math.pow(10, 9)) / 4;
        else return 0;
    }

    public static double logistic(double val) {
        return 1 / (Math.exp(val) + 2 + Math.exp(- val));
    }

    public static double sigmoid(double val) {
        return 2 / (Math.PI * (Math.exp(val) + Math.exp(- val)));
    }

    public static int fixed(int queryNum, double[][] train, int[] values, double d, int classesNum,
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

    public static int variable(int queryNum, double[][] train, int[] values, double k, int classesNum,
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

    public static double fmeasure(int[][] cm) {
        int k = cm.length;
        int totalSum = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                totalSum += cm[i][j];
            }
        }
        double precision = 0;
        double recall = 0;
        for (int i = 0; i < k; i++) {
            int fp = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fp += cm[i][j];
            }
            double weight = (double)(fp + cm[i][i])/totalSum;
            int fn = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fn += cm[j][i];
            }
            double pr = (cm[i][i] + fp == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fp);
            double re = (cm[i][i] + fn == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fn);
            precision += pr * weight;
            recall += re * weight;
        }
        return 2 * (precision * recall) / (precision + recall);
    }

    public static double fmeasureMicro(int[][] cm) {
        int k = cm.length;
        int totalSum = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                totalSum += cm[i][j];
            }
        }
        double precision = 0;
        double recall = 0;
        double fT = 0;
        for (int i = 0; i < k; i++) {
            int fp = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fp += cm[i][j];
            }
            double weight = (double)(fp + cm[i][i])/totalSum;
            int fn = 0;
            for (int j = 0; j < k; j++) {
                if (i != j) fn += cm[j][i];
            }
            double pr = (cm[i][i] + fp == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fp);
            double re = (cm[i][i] + fn == 0) ? 0 : (double)cm[i][i]/(cm[i][i] + fn);
            double f = (pr + re == 0) ? 0 : (2 * (pr * re)/(pr + re));
            fT += f * weight;
        }
        return fT;
    }

    public static int learnAndAnswerQuery(double[][] objects, int[] classes, int queryNum, int classesNum,
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
}
