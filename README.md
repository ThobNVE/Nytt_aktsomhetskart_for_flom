# Nytt_aktsomhetskart_for_flom
For oppbygging av grunnmodellen til NVEs nytt aktsomhetskart for flom.
Scripten ble testet i vassdrag nr. 122, "Gaula", Trondheims-region.

Scripten etableres snart, det er krav for å ha spesifikk mappesystem for det å fungere.
Mappesystem er som følges:
/HOVED MAPPE/
            > /BURNED/       # For DTM innbrent med elvdata som .tif filer
            > /envs/         # For Python miljø .yml (ikke nødvendig)
            > /FLOWACC/      # For Flow Accumulation .tif filer
            > /OUT/          # For DTM som er skalert til et valgt oppløsning
            > /RAW/          # For raw 1 m DTM data lastet ned fra geonorge
            > /Scripts/      # For scriptene, både .ipynb og .py
            > /Skalering/    # For test-scripts for skaleringsmetodikk (ikke nødvendig hvis skalerings-script eksisterer i /Scritps)

# Senere kan scriptene bransje ut med utvikling av bedre metodikk.

# Neste steg med grunn-scripten:
6. clean up code for clear paths
7. Bygg og inkludere flom-beregningene
8. Test med 10m data for flomkartlegging
9. Test med 1m data for flomkartlegging
10. Inkludere innsjøer.

# Hensyn til lagringsplass
Flere av filene i scripten trenger opp til 1GB lagringsplass per .tif fil. Videre, trenger "raw-dtm" mellom 10 - flere hundre GB lagringsplass for å laste ned i 1m oppløsning.
Ta hensyn på lagringsplassen deres, og jobbe med bare ETT vassdrag per brukssak.

# Det finnes flere filsti i denne scripten. De er som følges:
Raw 1m DTM filer:       RAW_PATH: "../RAW/"
                        RAW_DTM: f"{RAW_path}VD_{vassdrag_basin_name}_DTM.tif"
                    
Skalert DTM filer:      RS_PATH: "../OUT/"
                        RS_DTM: f"{RS_DIR}VD_{vassdrag_basin_name}_R{int(target_res)}m.tif"

Brennt DTM filer:       BN_PATH: "../BURNED/"
                        BN_DTM: f"{BN_PATH}VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m.tif"

Elvnett mask:           RNET_PATH: "../RNET/"
                        RNET_FIL: f"{RNET_PATH}VD{vassdrag_basin_name}_RB{burn_depth}_INV_R{int(target_res)}m.tif"

Fikset brennt DTM:      BN_FIX: f"{BN_PATH}VD_{vassdrag_basin_name}_RB{burn_depth}_FILL_R{int(target_res)}m.tif"
                        BN_FIX_RND = f"{FACC_path}VD_{vassdrag_basin_name}_RB{burn_depth}_FILL_RD_R{int(target_res)}m.tif"
                  
Flow accumulation:      FACC_PATH: "../FLOWACC/"
                        FACC_FIL: f"{FACC_PATH}FAC_VD_{vassdrag_basin_name}_RB{burn_depth}_FILL_RD_R{int(target_res)}m.tif"

