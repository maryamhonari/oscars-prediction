/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
//package oscarsextractor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Scanner;

/**
 *
 * @author alimashreghi
 */
public class OscarsExtractor {

    /**
     * @param line
     * @param pattern
     * @return 
     * 
     * 
     * 
     */
    
    public static String getTagValue(String line, String pattern){
       
        int x = line.indexOf(pattern);
        int i = x + pattern.length();
        while (i < line.length() && line.charAt(i) != '>') {
            i++;
        }
        i++;
        
        String result = "";
        while (i < line.length() && line.charAt(i) != '<') {
            result += line.charAt(i);
            i++;
        }
        return result;
    }
    
    public static String getTagValue(String line, String pattern, String exit){
       
        int x = line.indexOf(pattern);
        int i = x + pattern.length();
        while (i < line.length() && line.charAt(i) != '>') {
            i++;
        }
        i++;
        
        String result = "";
        while (i < line.length() && line.charAt(i) != '<' && !exit.contains("" + line.charAt(i))) {
            result += line.charAt(i);
            i++;
        }
        return result;
    }

    /**
     * Extracts information from search result of http://awardsdatabase.oscars.org/ampas_awards/BasicSearchInput.jsp
     * @param args
     * @throws Exception 
     */
    public static void main(String[] args) throws Exception {
        PrintStream out = new PrintStream(
                new FileOutputStream("all_info.txt"));

        //URL oracle = new URL("http://awardsdatabase.oscars.org/ampas_awards/DisplayMain.jsp?curTime=947756811356");

        //all_in.txt is the source of the search result page
        Scanner cin = new Scanner(new File("all_in.txt"));
        String winnerPattern = ">*<";
        String yearPattern = "searchLink&displayType=1&BSFromYear=";
        String nomineePattern = "searchLink&displayType=6&BSNominationID=";
        String filmPattern = "searchLink&displayType=3&BSFilmID";
        String nextCatPattern = "!@#$%^&*";
        
        String line;
        
        String nominee = "", film = "", year = "";
        
        String awards[] = 
        {"Actor Leading",
            "Actor Supporting",
            "Actress Leading",
            "Actress Supporting",
            "Animated",
            "Art Direction",
            "Cinematography",
            "Costume Design",
            "Directing",
            "Film Editing",
            "Foreign",
            "Makeup", 
            "Music Scoring", 
            "Music Song",
            "Best Picture",
            "Sound",
            "Sound Editing",
            "Visual Effects",
            "Writing"};

        
        boolean winnerFlag = false;
        boolean filmFound = false;
        boolean nomiFound = false;
        
        int cur = 0;
        
        while (cin.hasNext()) {
            line = cin.nextLine();
            
            if(line.contains(nextCatPattern)) cur++;
            
            if (line.contains(winnerPattern)){
                winnerFlag = true;
            }
            if(line.contains(nomineePattern)){
                nominee = getTagValue(line, nomineePattern);
                nomiFound = true;
            }
            if(line.contains(filmPattern)){
                film = getTagValue(line, filmPattern);
                filmFound = true;
            }
            
            if(filmFound && nomiFound){
                out.println(film + "|" + year + "|"+ awards[cur] + "|" + nominee + "|" + (winnerFlag ? "won" : "nom"));
                winnerFlag = filmFound = nomiFound = false;
            }
            
            if(line.contains(yearPattern)){
                year = getTagValue(line, yearPattern, " ");
            }
        }

    }
}
