import com.opencsv.CSVReader;
import javafx.util.Pair;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

public class SVM {

    private static Pair<int[][], double[]> train(int n, double[][] kernel, int[] target, int c, ArrayList<Integer> test) {
        double[] res = new double[n];
        double F = 0;
        for (int it = 0; it < 100; it++) {
            double oldF = F;
            for (int i = 0; i < n; i++) {
                if (test.contains(i)) continue;
                for (int j = i + 1; j < n; j++) {
                    if (test.contains(j)) continue;
                    double mu, muMin, a, z, L, R;
                    if (target[i] == target[j]) {
                        L = Math.max(-res[i], res[j] - c);
                        R = Math.min(c - res[i], res[j]);
                        muMin = 0;
                        for (int k = 0; k < n; k++) {
                            if (test.contains(k)) continue;
                            muMin += res[k] * (kernel[i][k] - kernel[j][k]);
                        }
                        a = (2 * kernel[i][j] - kernel[i][i] - kernel[j][j]);
                        z = muMin / a;
                    } else {
                        L = Math.max(-res[i], -res[j]);
                        R = Math.min(c - res[i], c - res[j]);
                        muMin = 0;
                        for (int k = 0; k < n; k++) {
                            if (test.contains(k)) continue;
                            muMin += res[k] * (kernel[i][k] + kernel[j][k]);
                        }
                        muMin -= 2;
                        a = (-2 * kernel[i][j] - kernel[i][i] - kernel[j][j]);
                        z = muMin / a;
                    }

                    if (a < 0) {
                        if (z >= L && z <= R) mu = z;
                        else {
                            if (z < L) mu = L;
                            else mu = R;
                        }
                        F += -0.5 * (2 * muMin * mu - a * mu * mu);
                    }
                    else if (a > 0) {
                        if (z >= L && z <= R) {
                            if (z - L > R - z) mu = L;
                            else mu = R;
                        }
                        else {
                            if (z < L) mu = R;
                            else mu = L;
                        }
                        F += -0.5 * (2 * muMin * mu - a * mu * mu);
                    } else {
                        muMin *= (-0.5);
                        if (muMin < 0) mu = L;
                        else mu = R;
                        F += 2 * muMin * mu;
                    }

                    if (target[i] == target[j]) {
                        res[i] += mu;
                        res[j] -= mu;
                    } else {
                        res[i] += mu;
                        res[j] += mu;
                    }
                }
            }
            if (F - oldF < 10e-70) break;
        }
        double b = 0;
        double trueB = 0;
        int cnt = 0;
        for (int i = 0; i < n; i++) {
            if (Math.abs(res[i]) < 10e-7) continue;
            cnt++;
            double bb = 0;
            for (int j = 0; j < n; j++) {
                bb += kernel[i][j] * res[j];
            }
            bb *= target[i];
            trueB += target[i] - bb;
            break;
        }
        b = trueB / cnt;
        int[][] conf = new int[2][2];
        for (Integer i: test) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += res[k] * kernel[i][k];
            }
            sum *= target[i];
            sum += b;
            conf[sum < 0 ? 0 : 1][(target[i] == -1 ? 0 : 1)]++;
        }
        double[] fres = new double[n + 1];
        System.arraycopy(res, 0, fres, 0, n);
        fres[n] = b;
        return new Pair<>(conf, fres);
    }

    private static double calcFMeasure(int samples, int batchSize, double[][] kernel, int[] classes, int c) {
        int[][] conf = new int[2][2];
        for (int i = 0; i < samples; i += batchSize) {
            ArrayList<Integer> test = new ArrayList<>();
            for (int j = i; j < i + batchSize; j++) {
                if (j == samples) break;
                test.add(j);
            }
            Pair<int[][], double[]> res = train(samples, kernel, classes, c, test);
            int[][] confB = res.getKey();
            conf[0][0] += confB[0][0];
            conf[0][1] += confB[0][1];
            conf[1][0] += confB[1][0];
            conf[1][1] += confB[1][1];
        }
        return Utils.fmeasure(conf);
    }

    private static void findOptimalForFile(String fileName) throws Exception {
        String[] tokens = fileName.split("/");
        String name = tokens[tokens.length - 1].split("\\.")[0];
        BufferedWriter log = new BufferedWriter(new FileWriter("l2og-" + name + ".txt"));
        {
            List<String[]> values = new CSVReader(new FileReader(fileName)).readAll();
            values.remove(0);  // header
            int samples = values.size();
            int attributes = values.get(0).length - 1;
            double[][] objects = new double[samples][attributes];
            int[] classes = new int[samples];
            for (int i = 0; i < samples; i++) {
                for (int j = 0; j < attributes; j++) {
                    objects[i][j] = Double.valueOf(values.get(i)[j]);
                }
                classes[i] = values.get(i)[attributes].equals("P") ? 1 : -1;
            }
            int batchSize = samples / 10;
            double[][] kernel = new double[samples][samples];
            // Linear
            {
                double maxF = 0;
                int maxC = 0;
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < samples; j++) {
                        kernel[i][j] = classes[i] * classes[j] * Utils.scalarProduct(objects[i], objects[j]);
                    }
                }
                for (int c = 1; c < 15; c++) {
                    double F = calcFMeasure(samples, batchSize, kernel, classes, c);
                    if (F > maxF) {
                        maxF = F;
                        maxC = c;
                    }
                    log.write("Linear kernel, c = " + c + ": F = " + F + "\n");
                    log.flush();
                }
                log.write("Linear kernel, optimal c = " + maxC + ": F = " + maxF + "\n");
                double[] linearFormula = train(samples, kernel, classes, maxC, new ArrayList<>()).getValue();
                for (double d : linearFormula) log.write(String.format("%.9f ", d));
                log.write("\n\n");
                log.flush();
            }
            //Polynomial with degree equal to p
            {
                double maxF = 0;
                int maxP = 0;
                int maxC = 0;
                for (int p = 2; p < 15; p++) {
                    for (int i = 0; i < samples; i++) {
                        for (int j = 0; j < samples; j++) {
                            kernel[i][j] *= Utils.scalarProduct(objects[i], objects[j]);
                        }
                    }
                    for (int c = 1; c < 15; c++) {
                        double F = calcFMeasure(samples, batchSize, kernel, classes, c);
                        if (F > maxF) {
                            maxF = F;
                            maxC = c;
                            maxP = p;
                        }
                        log.write("Polynomial with degree equal to p = " + p + " kernel, c = " + c + ": F = " + F + "\n");
                        log.flush();
                    }
                }
                log.write("Polynomial kernel, optimal p = " + maxP + ", c = " + maxC + ": F = " + maxF + "\n");
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < samples; j++) {
                        kernel[i][j] = 0;
                        kernel[i][j] = classes[i] * classes[j] * Utils.scalarProduct(objects[i], objects[j]);
                        for (int p = 1; p < maxP; p++) kernel[i][j] *= Utils.scalarProduct(objects[i], objects[j]);
                    }
                }
                double[] linearFormula = train(samples, kernel, classes, maxC, new ArrayList<>()).getValue();
                for (double d : linearFormula) log.write(String.format("%.9f ", d));
                log.write("\n\n");
                log.flush();
            }
            //Polynomial with degree less than p
            {
                double maxF = 0;
                int maxP = 0;
                int maxC = 0;
                for (int p = 1; p < 15; p++) {
                    for (int i = 0; i < samples; i++) {
                        for (int j = 0; j < samples; j++) {
                            double sc = Utils.scalarProduct(objects[i], objects[j]) + 1;
                            kernel[i][j] = classes[i] * classes[j] * Math.pow(sc, p);
                        }
                    }
                    for (int c = 1; c < 15; c++) {
                        double F = calcFMeasure(samples, batchSize, kernel, classes, c);
                        if (F > maxF) {
                            maxF = F;
                            maxC = c;
                            maxP = p;
                        }
                        log.write("Polynomial with degree less than p = " + p + " kernel, c = " + c + ": F = " + F + "\n");
                        log.flush();
                    }
                }
                log.write("Polynomial kernel (with degree less than), optimal p = " + maxP + ", c = " + maxC + ": F = " + maxF + "\n");
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < samples; j++) {
                        double sc = Utils.scalarProduct(objects[i], objects[j]) + 1;
                        kernel[i][j] = classes[i] * classes[j] * Math.pow(sc, maxP);
                    }
                }
                double[] linearFormula = train(samples, kernel, classes, maxC, new ArrayList<>()).getValue();
                for (double d : linearFormula) log.write(String.format("%.9f ", d));
                log.write("\n\n");
                log.flush();
            }
            //Radial
            {
                double maxF = 0;
                double maxB = 0;
                int maxC = 0;
                for (double b = 0.03; b < 1; b += 0.03) {
                    for (int i = 0; i < samples; i++) {
                        for (int j = 0; j < samples; j++) {
                            kernel[i][j] = classes[i] * classes[j] *
                                    Math.exp(-b * Math.pow(Utils.norm(Utils.subtract(objects[i], objects[j])), 2));
                        }
                    }
                    for (int c = 1; c < 15; c += 3) {
                        double F = calcFMeasure(samples, batchSize, kernel, classes, c);
                        if (F > maxF) {
                            maxF = F;
                            maxC = c;
                            maxB = b;
                        }
                        log.write("Radial kernel, b = " + b + ", c = " + c + ": F = " + F + "\n");
                        log.flush();
                    }
                }
                log.write("Radial kernel, optimal b = " + maxB + ", c = " + maxC + ": F = " + maxF + "\n");
                for (int i = 0; i < samples; i++) {
                    for (int j = 0; j < samples; j++) {
                        kernel[i][j] = classes[i] * classes[j] *
                                Math.exp(-maxB * Math.pow(Utils.norm(Utils.subtract(objects[i], objects[j])), 2));
                    }
                }
                double[] linearFormula = train(samples, kernel, classes, maxC, new ArrayList<>()).getValue();
                for (double d : linearFormula) log.write(String.format("%.9f ", d));
                log.write("\n\n");
                log.flush();
            }
        }
    }

    public static void main(String[] args) throws Exception {
        findOptimalForFile("src/main/resources/chips.csv");
        findOptimalForFile("src/main/resources/geyser.csv");
    }
}