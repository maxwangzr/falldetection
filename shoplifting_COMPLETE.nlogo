;Complete

extensions
[
  csv              ;needed to import terrain datafrom CSV file
  matrix           ;needed to work on matrices (2D tables)
  rnd              ;needed to generate a random number based on a probability distribution (weights)
]

__includes ["terrain.nls" "CRAVED_method.nls"]   ;contains the code used to generate the terrain //note: the variables must be defined under "globals"


globals
[
  ;colours of the patches
  colour_background
  colour_shelves
  colour_aisles
  colour_entrance
  colour_exit
  colour_outside
  colour_intersection

  ;regular customers
  interval_creation_RC

  ;shoplifter
  time_creation_SC

  ;guardians
  time_creation_RG
  N_RG

  ;colours of the turtles
  colour_walking_regularcustomer
  colour_standing_regularcustomer
  colour_walking_SC
  colour_standing_SC
  colour_walking_RG
  colour_standing_RG

  ;constant used to make browsing behaviour stochastic
  decide_browsetobuy_threshold
  decide_browsetosteal_threshold

  ;variables used to record what happened to the shoplifter
  has_been_detected
  has_noticed_detection
  ;added variables for CCTV and EAS simulation
  total_thefts          ; total number of thefts
  successful_escapes    ; number of successful escapes
  detected_thieves      ; number of thieves detected
  undetected_thieves    ; number of thieves not detected
  cctv_detections       ; number of CCTV detections
  eas_detections        ; number of EAS detections
  cctv_range            ; range of CCTV cameras
  eas_coverage          ; coverage of EAS tagging
  output_file           ; output file name for results
  experiment-mode       ; current mode (Only_N_RG, CCTV_Only, EAS_Only, CCTV_and_EAS)


  stop_simulation?

]

turtles-own
[
  init_time              ;time when turtle is created
  role                   ;role of the turtle (customer, guardian)
  shoplifter             ;criminal property of the turtle (shoplifter Y N)
  state                  ;state of the turtle re its movement (walking, standing)
  leaving                ;state of the turtle - turns to "Y" when the turtle has to leave the supermarket
  _candidates            ;the patch-set of neighbouring patches where the turtle can go
  pause_timer            ;timer used to make the turtle stand in one place for a few ticks (looks at the shelves)
  pause_timer_max        ;max value for timer
  number_of_products     ;number of products in the turtle's shopping basket
  number_of_products_max ;number of products the turtle has to take before leaving
  floorplan              ;each turtle has a mental map of the floorplan where they record the number of times they visited each patch
  Ntimes_visited         ;constant used to record how many times a patch has been visited
  x_candidates           ;X coordinates of the neighbouring patches
  y_candidates           ;y coordinates of the neighbouring patches
  index_bestpatch        ;variable needed to select next patch when a turtle is at an intersection
  number_of_stolen_products_max ;number of products the shoplifter intends to steal before leaving
  number_of_stolen_products ;number of products that the shoplifter has stolen
]

patches-own
[
  category              ;type of patch (shelves, aisles, intersections, entrance, exit, outside)
  CRAVED                ;six-digit criminogenic index of the products on a given patch
]

;;---------------------------
to setup
 clear-all

 create_supermarket

 add_products

 ;define properties of turtles
 set interval_creation_RC 10
 set time_creation_SC 500
 set time_creation_RG 0
 set N_RG 4

 ;define colours of the turtles
 set colour_walking_regularcustomer grey
 set colour_standing_regularcustomer violet
 set colour_walking_SC green
 set colour_standing_SC red
 set colour_walking_RG blue
 set colour_standing_RG blue

 ;define constants for the behaviour of the turtles
 set decide_browsetobuy_threshold 10
 set decide_browsetosteal_threshold 25
 ; Initialize variables
 set total_thefts 0
 set successful_escapes 0
 set detected_thieves 0
 set undetected_thieves 0
 set cctv_detections 0
 set eas_detections 0

 reset-ticks
end

;;-------------------------------------------------------
;; MAIN PROCEDURE
to go

  ;create new agents (customers) in the simulation
  if (ticks mod interval_creation_RC = 0) [make-turtles "RC" 2]

  ;create new agents (shoplifters) in the simulation
  if (ticks = time_creation_SC) [make-turtles "SC" 1]

  ;create new agents (guardians) in the simulation
  if (ticks = time_creation_RG) [make-turtles "RG" N_RG]

  ;manage the behaviour of agents during the simulation
  ask turtles
  [

    ;;-------------
    (ifelse
    ;Behaviour when the agent walks along the aisles of the supermarket
    [category] of patch-here = "aisles"
    ;note: ANOTHER WAY WOULD BE TO WRITE: ifelse pcolor = colour_aisles
      [
       ;if the turtle is in the 'walking' state
       ifelse state = "walking"
       [
          ;verify if customer is a shoplifter or not
          ifelse shoplifter = "N"
          [
          ;call procedure to decide to browse to buy
          decide_browsetobuy
          ;move forward one patch
          if state != "browsingtobuy" [fd 1]
          ]
          [
          ;call procedure to decide to browse to steal
          decide_browsetosteal
          ;move forward one patch
          if state != "browsingtosteal" [fd 1]
          ]
       ]
       [
          ;if the turtle is not in the 'walking' state (i.e., if it is in the standing state)

        ;do nothing for #pause_timer ticks
        set pause_timer pause_timer - 1

        ;when it reaches the end of the countdown
        if pause_timer = 1
        [
         ; re-initiatilize counter
         set pause_timer pause_timer_max

         ;turtle returns to 'walking' state
         set state "walking"

          if shoplifter = "N" and role = "customer"
            [
            ;increment the number of products in the shopping basket
            set number_of_products number_of_products + 1

            ;if number of products (stolen products or in the shopping basket) reaches the max limit then indicate that the customer should leave the store
            if (number_of_products >= number_of_products_max) [set leaving "Y"]

            ;turtle returns to its original colour
            set color colour_walking_regularcustomer
            ]
           if shoplifter = "Y"
            [
            ;increment the number of products in the shopping basket
            set number_of_stolen_products number_of_stolen_products + 1

            ;if number of products (stolen products or in the shopping basket) reaches the max limit then indicate that the customer should leave the store
            if (number_of_stolen_products = number_of_stolen_products_max) [set leaving "Y"]

            ; Call the detection procedure here
            detection

            ;turtle returns to its original colour
            set color colour_walking_SC
            ]


        ]
      ]
    ]

    ;;-------------
    ;Behaviour when the agent reaches an intersection
    [category] of patch-here = "intersection"
      [

      set _candidates patches with [distance myself = 1 and pcolor = colour_aisles  and not (pxcor = 1 and pycor = 24)]

      set x_candidates []
      set y_candidates []
      set Ntimes_visited []

      ;WARNING - everytime we read an agent-set it comes in a random order.
      ;So if we try to extract the x and y coordinates directly from the agent-set, we read pxcor for one patch and pycor for another patch!
      ;To set the agent-set in a fixed order, we use the command sort which creates a sorted list of agents.
      ;Important: the resulting list is no longer an agent-set so we cannot apply useful commands like "one-of")


        foreach sort _candidates
        [
          the-patch ->

            set x_candidates lput ([pxcor] of the-patch) x_candidates
            set y_candidates lput ([pycor] of the-patch) y_candidates

        ]


          ;create a list with the number of times the neighbouring patches have been visited
          let i 0
          repeat length x_candidates
          [
            set Ntimes_visited lput matrix:get floorplan item i y_candidates item i x_candidates Ntimes_visited
            set i i + 1
          ]

          let w []

          ;create weights to attractiveness of individual patches and randomly select one of them
          set w generate-weights Ntimes_visited


          let list_index n-values (count _candidates) [ k -> k ]

          let pairs (map list list_index w)

            ;report the first item of the pair selected using
            ;the second item (i.e., `last p`) as the weight
            set index_bestpatch first rnd:weighted-one-of-list pairs [ [p] -> last p ]

          face patch item index_bestpatch x_candidates item index_bestpatch y_candidates

          fd 1



      ]

    ;;-------------
    ;behaviour when the agent gets to the entrance
    ;if (pcolor = colour_entrance)
    [category] of patch-here = "entrance" [
      ;define a patch-set with relevant neighbouring patches (the entrance patch is removed from the patch-set so the agent cannot exit through the entrance)
      set _candidates patches with [distance myself = 1 and pcolor = colour_aisles  and not (pxcor = 1 and pycor = 24)]
      ;the turtle faces one of the patches and move forward by one patch
      face one-of _candidates
      fd 1
    ]

    ;;-------------
    ;behaviour when the agent gets to the exit
    [category] of patch-here = "exit" [
      ifelse leaving = "N"
        [set heading one-of [180 270]]
        [set heading 0]
      fd 1
    ]

    ;;-------------
    ;behaviour when the agent gets to the road
    pcolor = colour_outside [ die ]
  )

    ;update the mental map of turtles
    matrix:set floorplan pycor pxcor (matrix:get floorplan pycor pxcor + 1)

  ]

  tick

  if stop_simulation? = true
  [
    file-open output-file
    file-print (word "N_RG," N_RG ",CCTV Range," cctv_range ",EAS Coverage," eas_coverage)
    file-print (word "Total Thefts," total_thefts ",Successful Escapes," successful_escapes)
    file-print (word "Detected Thieves," detected_thieves ",Undetected Thieves," undetected_thieves)
    file-print (word "CCTV Detections," cctv_detections ",EAS Detections," eas_detections)
    file-close
    stop
  ]

end
;;-------------------------------------------------------
;; CREATE THE TERRAIN

to create_supermarket
  set_colour_patches
  add_patches
end

;;-------------------------------------------------------
;; ADD PRODUCTS

;assign a CRAVED index from a CSV file to each shelves patch
to add_products
   assign_CRAVED
end


;;-------------------------------------------------------
;; CREATE THE AGENTS

to make-turtles [turtle_category how_many] ;
; This procedure creates three types of agents:
; RC: regular customer - SC: shoplifter customer - RG: regular guardian

  ;;set default shape of agents
  set-default-shape turtles "person"

  ;;create agents at the start
  create-turtles how_many [

    ;; record when they were created
    set init_time ticks

    ;; set the states of the turtle
    set state "walking"
    set leaving "N"

    if turtle_category = "RC" ; regular customer
    [
     set role "customer"
     set shoplifter "N"
     set color colour_walking_regularcustomer

    ;; initialise their shopping basket
    set number_of_products 0
    set number_of_products_max 3 + random 3
    ]

    if turtle_category = "SC" ; shoplifter-customer
    [
     set role "customer"
     set shoplifter "Y"
     set color colour_walking_SC
     set shape "person soldier"
     set number_of_stolen_products 0
     set number_of_stolen_products_max 4 + random 2
    ]

    if turtle_category = "RG" ; regular guardian
    [
     set role "guardian"
     set shoplifter "N"
     set color colour_walking_RG
     set shape "person police"
    ]

    ;define how long the turtle will stay
    set pause_timer_max 6 + random 5

    ;; place the agent facing downward at the entrance
    setxy 1 23
    set heading 180

    ;; give agents a mental map of where they went in the supermarket
    set floorplan matrix:make-constant max-pycor max-pxcor 0

  ]
end

;;-------------------------------------------------------
;; BEHAVIOUR OF CUSTOMERS (Browsing to buy procedures)

to browsetobuy
 ifelse role = "customer" [set color colour_standing_regularcustomer] [set color colour_standing_RG]
 set pause_timer pause_timer_max
 set state "browsingtobuy"
end

to decide_browsetobuy
   if (random 100 <= decide_browsetobuy_threshold) [browsetobuy]
end

;;-------------------------------------------------------
;; BEHAVIOUR OF SHOPLIFTERS (Browsing to steal procedures)

to decide_browsetosteal
  if (random 100  <= decide_browsetosteal_threshold) [browsetosteal]
end

to browsetosteal
 set color colour_standing_SC
 set pause_timer pause_timer_max
 set state "browsingtosteal"
end

;;-------------------------------------------------------
;; Weighted navigation through the supermarket - Probability of choosing a patch

to-report generate-weights [n_visits]
;the reporter is used to operationalise the following principle:
;the more times a (neighbouring) patch has been visited by an agent, the less likely the agent is to move to this patch.
;the reporter takes the number of times neighbouring patches have been visited (by the agent) and returns normalised individual weights where
;each of them corresponds to the probability that a given patch is selected by the agent

  let omega n-values (length n_visits) [1 + sum n_visits]

  let numeratorWk (map - omega n_visits)

  let Wk (map / numeratorWk omega)
  ; normalise Wk
  set Wk (map / Wk n-values (length n_visits) [sum(Wk)])

  report Wk

end
;;-------------------------------------------------------
;; Procedures for detection
to-report whats_in_between [turtle_1 turtle_2]

  ;this reporter counts how many patches and how many turtles there are between two patches
  ;if the two turtles are not in line of sight it reports [99999 99999]
  ;SIMPLIFICATION: because of the layout, we know two turtles can only see each other if they are in the same row or in the same column

  let region[]               ;portion of space between the two turtles
  let number_patches 99999   ;default value when turtles are not in line of sight
  let number_turtles 99999   ;default value when  turtles are not in line of sight
  let result[]               ;the output

  ;extract the coordinates of the two turtles
  let x1 [pxcor] of turtle_1
  let y1 [pycor] of turtle_1
  let x2 [pxcor] of turtle_2
  let y2 [pycor] of turtle_2

  ;if the turtles are in the same column
  ifelse (x1 = x2)
  [
    ;define patch-set of interest
    ifelse (y1 < y2 )
      [set region patches with [ pxcor = x1 and (pycor >= y1 and pycor <= y2)]]  ;turtle 2 above turtle 1
      [set region patches with [ pxcor = x1 and (pycor >= y2 and pycor <= y1)]]  ;turtle 1 above turtle 2

      if sort region = sort (region with [pcolor = colour_aisles or pcolor = colour_intersection ])
      [
      ;count number of patches between the two turtles
      set number_patches abs (y2 - y1)
      ;count number of turtles between the two turtles
      set number_turtles count turtles-on region - 2
      ]
  ]
  [ ;if the turtles are in the same row
    if (y1 = y2)
    [
      ;define patch-set of interest
      ifelse ( x1 < x2 )
        [set region patches with [ pycor = y1  and (pxcor >= x1 and pxcor <= x2) ]]  ;turtle 2 on the right of turtle 1
        [set region patches with [ pycor = y1  and (pxcor >= x2 and pycor <= x1) ]]  ;turtle 1 on the right of turtle 2

      if sort region = sort (region with [pcolor = colour_aisles or pcolor = colour_intersection ])
      [
      ;count number of patches between the two turtles
      set number_patches abs (x2 - x1)
      ;count number of turtles between the two turtles
      set number_turtles count turtles-on region - 2
      ]
    ]
  ]

  set result list number_patches number_turtles

  report result

end

to-report probability_detection [in_between prob]
;this reporter calculates the probability that a turtle notices that the other turtle has concealed an item
;the reporter requires a list with two values as an input: number of patches and number of turtles
;see reporter "whats_in_between"

  let number_patches item 0 in_between
  let number_turtles item 1 in_between
  let probability prob ^ (1 + number_patches + number_turtles)     ;main equation for the detection

  report probability

end

to-report binomial_dist [prob]
;this reporter is used to randomly select 1 or 0 with 'prob' representing Pr(1)

  let list_index [0 1]
  let prob_list list (1 - prob) prob
  let pairs (map list list_index prob_list)                      ; create pairs from two lists - http://ccl.northwestern.edu/netlogo/docs/dict/map.html
  report first rnd:weighted-one-of-list pairs [ [p] -> last p ]  ; https://ccl.northwestern.edu/netlogo/docs/rnd.html
  ;to understand the above, paste this line:
  ;repeat 100 [type first rnd:weighted-one-of-list [ [ "A" 0.1 ] [ "B" 0.9 ] ] [ [p] -> last p ]]
end

to detection

  let the_shoplifter self
  set has_been_detected 0
  set has_noticed_detection 0
  ;; CCTV detect logic
  if cctv [
    print "CCTV is active."
    ; if the shoplifter is in the range of cctv then tag as detected
    if any? cctv_positions with [distance the_shoplifter <= cctv_range] [
      set has_been_detected has_been_detected + 1
      set cctv_detections cctv_detections + 1
      print (word "CCTV detected thief at " [pxcor] of the_shoplifter ", " [pycor] of the_shoplifter)
    ]
  ]

  ;; EAS detect logic
  if eas [
    print "EAS is active."
    ; if the shoflifter take the product and go through the exit then EAS alarm
    if number_of_stolen_products > 0 and [category] of patch-here = "exit" [
      set eas_detections eas_detections + 1
      print (word "EAS alarm triggered by thief at exit! Thief detected with " number_of_stolen_products " stolen items.")
      ; shoplifter drop the product
      set number_of_stolen_products 0
    ]
  ]

  foreach sort turtles with [role = "guardian"]
  [
    x ->
    let the_guard x

    ;determine if guardian saw the shoplifting act
    if (binomial_dist (probability_detection whats_in_between the_guard the_shoplifter prob_guardians) = 1)
    [
      set has_been_detected has_been_detected + 1

      ;determine if shoplifter noticed guardian saw the shoplifting act
      if (binomial_dist (probability_detection whats_in_between the_shoplifter the_guard prob_shoplifters) = 1)
      [
        set has_noticed_detection has_noticed_detection + 1
      ]
    ]
  ]

  ifelse has_been_detected = 0
  [
    set undetected_thieves undetected_thieves + 1
    print "Guardians did not see the shoplifting act"
    if (leaving = "Y") [ set
      set successful_escapes successful_escapes + 1
      stop_simulation? true ]
  ]
  [
   print "At least one guardian saw the shoplifting act"
   set total_thefts total_thefts + 1


  ;whether they noticed a guard saw the shoplifting act
   ifelse (has_noticed_detection = 0)
   [
    print "The shoplifter did not notice they have been detected. The guardian caught them."
    set stop_simulation? true
   ]
   [
    set number_of_stolen_products 0
    set leaving "Y"
    set stop_simulation? true
    print "The shoplifter did notice it. They put all products back on the shelves and left the store."
   ]
  ]

end
;-------------------------------------------
@#$#@#$#@
GRAPHICS-WINDOW
159
10
565
417
-1
-1
12.84
1
9
1
1
1
0
1
1
1
0
30
0
30
0
0
1
ticks
30.0

BUTTON
46
39
116
72
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
46
82
117
115
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
46
125
124
158
Go-once
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
159
444
331
477
prob_guardians
prob_guardians
0
1
1.0
0.001
1
NIL
HORIZONTAL

SLIDER
160
489
332
522
prob_shoplifters
prob_shoplifters
0
1
0.803
0.001
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

person police
false
0
Polygon -1 true false 124 91 150 165 178 91
Polygon -13345367 true false 134 91 149 106 134 181 149 196 164 181 149 106 164 91
Polygon -13345367 true false 180 195 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285
Polygon -13345367 true false 120 90 105 90 60 195 90 210 116 158 120 195 180 195 184 158 210 210 240 195 195 90 180 90 165 105 150 165 135 105 120 90
Rectangle -7500403 true true 123 76 176 92
Circle -7500403 true true 110 5 80
Polygon -13345367 true false 150 26 110 41 97 29 137 -1 158 6 185 0 201 6 196 23 204 34 180 33
Line -13345367 false 121 90 194 90
Line -16777216 false 148 143 150 196
Rectangle -16777216 true false 116 186 182 198
Rectangle -16777216 true false 109 183 124 227
Rectangle -16777216 true false 176 183 195 205
Circle -1 true false 152 143 9
Circle -1 true false 152 166 9
Polygon -1184463 true false 172 112 191 112 185 133 179 133
Polygon -1184463 true false 175 6 194 6 189 21 180 21
Line -1184463 false 149 24 197 24
Rectangle -16777216 true false 101 177 122 187
Rectangle -16777216 true false 179 164 183 186

person service
false
0
Polygon -7500403 true true 180 195 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285
Polygon -1 true false 120 90 105 90 60 195 90 210 120 150 120 195 180 195 180 150 210 210 240 195 195 90 180 90 165 105 150 165 135 105 120 90
Polygon -1 true false 123 90 149 141 177 90
Rectangle -7500403 true true 123 76 176 92
Circle -7500403 true true 110 5 80
Line -13345367 false 121 90 194 90
Line -16777216 false 148 143 150 196
Rectangle -16777216 true false 116 186 182 198
Circle -1 true false 152 143 9
Circle -1 true false 152 166 9
Rectangle -16777216 true false 179 164 183 186
Polygon -2674135 true false 180 90 195 90 183 160 180 195 150 195 150 135 180 90
Polygon -2674135 true false 120 90 105 90 114 161 120 195 150 195 150 135 120 90
Polygon -2674135 true false 155 91 128 77 128 101
Rectangle -16777216 true false 118 129 141 140
Polygon -2674135 true false 145 91 172 77 172 101

person soldier
false
0
Rectangle -7500403 true true 127 79 172 94
Polygon -10899396 true false 105 90 60 195 90 210 135 105
Polygon -10899396 true false 195 90 240 195 210 210 165 105
Circle -7500403 true true 110 5 80
Polygon -10899396 true false 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Polygon -6459832 true false 120 90 105 90 180 195 180 165
Line -6459832 false 109 105 139 105
Line -6459832 false 122 125 151 117
Line -6459832 false 137 143 159 134
Line -6459832 false 158 179 181 158
Line -6459832 false 146 160 169 146
Rectangle -6459832 true false 120 193 180 201
Polygon -6459832 true false 122 4 107 16 102 39 105 53 148 34 192 27 189 17 172 2 145 0
Polygon -16777216 true false 183 90 240 15 247 22 193 90
Rectangle -6459832 true false 114 187 128 208
Rectangle -6459832 true false 177 187 191 208

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="shoplifting_experiment" repetitions="1000" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <timeLimit steps="5000"/>
    <metric>has_been_detected</metric>
    <metric>has_been_caught</metric>
    <metric>has_noticed_detection</metric>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
