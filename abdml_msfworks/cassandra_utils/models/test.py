        # Extract `active` from componentBatchList
        component_batch_list = get_attr(main_message, "componentBatchList")
        if component_batch_list and isinstance(component_batch_list, list):
            active_flag = any(item.get("act") == "True" for item in component_batch_list)
        else:
            active_flag = False


        #customer extraction
        output_value_list = get_attr(main_message,"outputValueList") 
        if output_value_list and isinstance(output_value_list,list):
            customer = any(item.get("cust") for item in output_value_list)
        else:
            customer = None

#produced extr
        producted_stock_list =  get_attr(main_message,"produceList")
            if producted_stock_list and isinstance(producted_stock_list, list)
                pr_stk_id = any(item.get("stId") for item in producted_stock_list)
                pr_stk_name = any(item.get("stNm") for item in producted_stock_list)
                pr_stl_numb = any(item.get("stNo") for item in producted_stock_list)

        year = measurement_date.strftime("%Y")