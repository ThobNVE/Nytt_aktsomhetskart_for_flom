# Nytt_aktsomhetskart_for_flom
For oppbygging av grunnmodellen til NVEs nytt aktsomhetskart for flom.
Scripten ble testet i vassdrag nr. 122, "Gaula", Trondheims-region.

Før du begynne å bruke, må du:
1. Oppsett et Anaconda-python system.
2. Bygg conda miljøet med Python_GIS.yml (i /envs/)
3. Aktivere Python_GIS miljøet (i powershell hvis du bruker jupyter, eller i VS Studio/ annet Python programvare)

Det er krav for å ha spesifikk mappesystem for det å fungere.
Mappesystem er som følges:
>/HOVED MAPPE/

>            > /BURNED/       # For DTM innbrent med elvdata som .tif filer

>            > /ELV/          # For Elvenettdata produsert av DTM.

>            > /FLD/          # Flomområder

>            > /RNET/         # Invertert elvenettverk for innbrenning og geometri endring

>            > /FLOWACC/      # For Flow Accumulation .tif filer

>            > /OUT/          # For DTM som er skalert til et valgt oppløsning

>            > /RAW/          # For raw 1 m DTM data lastet ned fra geonorge

>            > /FDIR/         # For strømretning data 

>            > /Scripts/      # For scriptene, både .ipynb og .py

>            > /Skalering/    # For test-scripts for skaleringsmetodikk (ikke nødvendig hvis skalerings-script eksisterer i /Scritps)

>            > /envs/         # For Python miljø .yml (ikke nødvendig)

# Senere kan scriptene bransje ut med utvikling av bedre metodikk.

# Neste steg med grunn-scripten:
2.	Pyflwdir – sensitivity analyse
    2.1.	Gjentaksinterval
  	
    2.2.	B verdi for 1:200
  	
    2.3.	Forskjellige områder
  	
    2.4.	Verdi for «vannfylker»
  	
    2.5.	HAND method investigere
  	
4.	Bruk pyflwdir opp til minimal area subbasins celle og prøve:
    3.1.	Tverrprofiler metode
  	
    3.2.	Differential VSS over område – «VSS verdi» reduserer over hvert cell fra opprinelse – kan ble enkelt for å inkludere infiltrasjon senere
  	
    3.3.	Kernel method – litt det samme men med kernel og vekt av reduseringsverdi med «helning» som vekt.
  	
    3.3.1.	HAND method? Må investigere HAND = tverrprofiler.
6.	Les gjennom dokumentene
7.	Autoroute testing. – Petra
8.	
    5.1.	LISFLOOD & AUTOROUTE

# Hensyn til lagringsplass
Flere av filene i scripten trenger opp til 1GB lagringsplass per .tif fil. Videre, trenger "raw-dtm" mellom 10 - flere hundre GB lagringsplass for å laste ned i 1m oppløsning.
Ta hensyn på lagringsplassen deres, og jobbe med bare ETT vassdrag per brukssak.

# Det finnes flere filsti i denne scripten. De er som følges:
Raw 1m DTM filer:       
>RAW_PATH: "../RAW/"
                        
>RAW_DTM: f"{RAW_path}VD_{vassdrag_basin_name}_DTM.tif"
                    
Skalert DTM filer:      
>RS_PATH: "../OUT/"
                        
>RS_DTM: f"{RS_DIR}VD_{vassdrag_basin_name}_R{int(target_res)}m.tif"

Brennt DTM filer:       
>BN_PATH: "../BURNED/"
                        
>BN_DTM: f"{BN_PATH}VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m.tif"

Elvnett mask:           
>RNET_PATH: "../RNET/"
                        
>RNET_FIL: f"{RNET_PATH}VD{vassdrag_basin_name}_RB{burn_depth}_INV_R{int(target_res)}m.tif"

Fikset brennt DTM:      
>BN_FIX: f"{BN_PATH}VD_{vassdrag_basin_name}_RB{burn_depth}_FILL_R{int(target_res)}m.tif"
                        
>BN_FIX_RND = f"{FACC_path}VD_{vassdrag_basin_name}_RB{burn_depth}_FILL_RD_R{int(target_res)}m.tif"
                  
Flow accumulation:      
>FACC_PATH: "../FLOWACC/"

>FACC_FIL: f"{FACC_PATH}FAC_VD_{vassdrag_basin_name}_RB{burn_depth}_FILL_RD_R{int(target_res)}m.tif"

Strahler Stream Order:
>STRAHLER_PATH = "../ELV/"

>STRAHLER_FIL = f"{STRAHLER_PATH}SORD_VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m.tif"

Strømretning
>FDIR_PATH = "../FDIR"

FDIR_FIL = f"{FDIR_PATH}FDIR_VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m.tif"

Flom-output (ENKELT)
>FLD_PATH = "../FLD"

>FLD_FIL_ELVIS = f"{FLD_PATH}FLD_{b}_VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m_ELVIS.tif"

>FLD_FIL_STREAMS = f"{FLD_PATH}FLD_{b}_VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m_STREAMS.tif"

>FLD_FIL_ELVIS_VEC = f"{FLD_PATH}FLD_{b}_VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m_ELVIS.gpkg"

>FLD_FIL_STREAMS_VEC = f"{FLD_PATH}FLD_{b}_VD_{vassdrag_basin_name}_RB{burn_depth}_R{int(target_res)}m_STREAMS.gpkg"


