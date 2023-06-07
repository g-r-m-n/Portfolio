-- Using Postgre SQL syntax 


-- SQL query that displays the number of customers per industry
-- Assumption: CUSTOMER_ID is a unique key in the table Customer.
Select 
INDUSTRY,
count(CUSTOMER_ID) as number_of_customers
From Customer
Group by 1;


-- SQL query that displays the average invoice total per industry
-- Additional assumption: SALE_ID is the unique id for a sales incident in table Sales.  

Select 
INDUSTRY,
sum(INVOICE_TOTAL) as INVOICE_TOTAL
From Sales 
join Customer using CUSTOMER_ID
Group by 1;


-- SQL query that displays what each customer spent per month, if that value is bigger than 100.
Select
CUSTOMER_ID,
DATE_PART(month, DATE)  as Month,   
-- In case DATE is stored as string, use the following line instead of the previous one: 
-- DATE_PART(month, TO_DATE(DATE,'YYYY/MM/DD') ) as Month, 
sum(INVOICE_TOTAL) as INVOICE_TOTAL
From Sales 
join Customer using CUSTOMER_ID
Group by 1, 2
Having sum(INVOICE_TOTAL) > 100 -- get only cases where a customer has an invoice total more than 100.
ORDER BY 1, 2 -- order the results by customer id and month in ascending order.
;