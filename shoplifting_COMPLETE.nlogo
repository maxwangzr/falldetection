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
  
  total_thefts ; 总偷窃次数
  successful_escapes ; 成功偷窃并逃离次数
  detected_thieves ; 被守卫检测到的偷窃者数量
  undetected_thieves ; 未被检测的偷窃者数量
  total_simulation_time ; 模拟总运行时间
  escape_times ; 记录每个偷窃者的逃离时间
  
  cctv_positions ; 保存CCTV安装的位置

  cctv_detections ; 记录CCTV检测次数
  
  eas_positions ; 带有EAS标签的商品位置
  eas_detections ; EAS触发报警的次数
  output_file ; 当前实验的输出文件名


  
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
to set-experiment-output
  if experiment-mode = "Only_N_RG" [
    set output_file "Only_N_RG.csv"
  ]
  if experiment-mode = "CCTV_Only" [
    set output_file "CCTV_Only.csv"
  ]
  if experiment-mode = "EAS_Only" [
    set output_file "EAS_Only.csv"
  ]
  if experiment-mode = "CCTV_and_EAS" [
    set output_file "CCTV_and_EAS.csv"
  ]
end
;;---------------------------
to setup
 clear-all

 create_supermarket

 add_products

 ;define properties of turtles
 set interval_creation_RC 10
 set time_creation_SC 500
 set time_creation_RG 0
 set N_RG N_RG

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
  
 set total_thefts 0
 set successful_escapes 0
 set detected_thieves 0
 set undetected_thieves 0
 set total_simulation_time 0
 set escape_times []
 set cctv_detections 0
 set eas_detections 0 ; 初始化报警次数为0

 set cctv_range cctv_range ; 从 Slider 获取 CCTV 范围
 set eas_coverage eas_coverage ; 从 Slider 获取 EAS 覆盖比例

 set cctv_positions patches with [pxcor mod 5 = 0 and pycor mod 5 = 0] ; 每5个patch放一个CCTV（可根据需要调整）
 set eas_positions patches with [category = "shelves" and random 100 < eas_coverage] ; 随机为20%的货架商品添加标签

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
    ; behaviour when the agent gets to the exit
    [category] of patch-here = "exit" [
      if leaving = "Y" [
        if eas [
          ; 检查偷窃者是否携带带标签商品
          if number_of_stolen_products > 0 [
            set eas_detections eas_detections + 1 ; 增加报警次数
            print (word "EAS alarm triggered by thief at exit! Thief detected with " number_of_stolen_products " stolen items.")
            ; 偷窃者放弃商品
            set number_of_stolen_products 0
          ]
        ]
        ; 离开商店逻辑
        set heading 0
      ] 
        ; 非离开的偷窃者随机转向
        set heading one-of [180 270]
      
      ; 前进一步
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
  set total_simulation_time ticks ; 更新模拟时间
  ; 增加步数限制，确保模拟运行足够长时间
  if ticks >= 1000 [
    set stop_simulation? true
  ]
  if stop_simulation? = true
  [
    print "Simulation completed."
    print (word "Total Ticks (Simulation Steps): " ticks) ; 输出总步数
    print (word "Total Thefts: " total_thefts)
    print (word "Successful Escapes: " successful_escapes)
    print (word "Detected Thieves: " detected_thieves)
    print (word "Undetected Thieves: " undetected_thieves)
    print (word "Average Escape Time: " (ifelse-value (length escape_times = 0) [0] [mean escape_times]))
    print (word "CCTV Detections: " cctv_detections)
    print (word "EAS Detections: " eas_detections) ; 输出 EAS 报警次数
    set-experiment-output
    ; 如果文件不存在，写入标题
    if not file-exists? output_file [
      file-open output_file
      file-print "Experiment Mode,N_RG,CCTV Range,EAS Coverage,Total Thefts,Successful Escapes,Detected Thieves,CCTV Detections,EAS Detections,Simulation Steps"
      file-close
    ]

    ; 写入实验结果
    file-open output_file
    file-print (word experiment-mode "," N_RG "," cctv_range "," eas_coverage "," total_thefts "," successful_escapes "," detected_thieves "," cctv_detections "," eas_detections "," ticks)
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
 set total_thefts total_thefts + 1 ; 增加偷窃次数
 set number_of_stolen_products number_of_stolen_products + 1 ; 增加偷窃次数

  
 ; 检查是否窃取了带标签商品
 if [category] of patch-here = "shelves" and member? patch-here eas_positions [
   print (word "Thief stole a tagged item at " [pxcor] of patch-here ", " [pycor] of patch-here)
 ]

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
  
  if eas [
    print "EAS is active."
    if number_of_stolen_products > 0 [
      set eas_detections eas_detections + 1
      print (word "EAS alarm triggered by thief! Thief detected with " number_of_stolen_products " stolen items.")
      set number_of_stolen_products 0 ; 偷窃者被迫放弃商品
    ]
  ]

  ; 检查是否启用CCTV
  if cctv [
    print "CCTV is active."
    ; 如果偷窃者在CCTV范围内，直接标记为被检测
    if any? cctv_positions with [distance the_shoplifter <= cctv_range] [
      set has_been_detected has_been_detected + 1
      set cctv_detections cctv_detections + 1
      print (word "CCTV detected thief at " [pxcor] of the_shoplifter ", " [pycor] of the_shoplifter)
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
      set detected_thieves detected_thieves + 1 ; 记录被检测的偷窃者
      
      ;determine if shoplifter noticed guardian saw the shoplifting act
      if (binomial_dist (probability_detection whats_in_between the_shoplifter the_guard prob_shoplifters) = 1)
      [
        set has_noticed_detection has_noticed_detection + 1
      ]
    ]
  ]

  ifelse has_been_detected = 0
  [
    set undetected_thieves undetected_thieves + 1 ; 记录未被检测的偷窃者
    print "Guardians did not see the shoplifting act"
    if (leaving = "Y") [ 
      set successful_escapes successful_escapes + 1 ; 记录成功逃离次数
      set escape_times lput (ticks - init_time) escape_times ; 记录逃离时间
      set stop_simulation? true ]
  ]
  [
   print "At least one guardian saw the shoplifting act"
   set total_thefts total_thefts + 1 ; 总偷窃次数

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
