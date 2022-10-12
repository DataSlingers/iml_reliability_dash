import dash_bootstrap_components as dbc
def Navbar():
    navbar = dbc.NavbarSimple(
           children=[
                dbc.NavItem(dbc.NavLink("How to use", href="/instruction")),

#                 dbc.DropdownMenu(
#                  nav=True,
#                  in_navbar=True,
#                  label="Feature Importance",
#                  children=[
#                         dbc.DropdownMenuItem("Classification",href="/feature_importance_classification"),
#                         dbc.DropdownMenuItem("Regression",href="/feature_importance_regression"),
#                           ],
#                       ),
                    
              dbc.NavItem(dbc.NavLink("Feature Importance (Classification)", href="/feature_importance_classification")),
              dbc.NavItem(dbc.NavLink("Feature Importance (Regression)", href="/feature_importance_regression")),
              dbc.NavItem(dbc.NavLink("Clustering", href="/clustering")),
              dbc.NavItem(dbc.NavLink("Dimension Reduction (Clustering)", href="/dimension_reduction_clustering")),
              dbc.NavItem(dbc.NavLink("Dimension Reduction (Local Neighbors)", href="/knn")),
#                 dbc.DropdownMenu(
#                  nav=True,
#                  in_navbar=True,
#                  label="Dimension Reduction",
#                  children=[
#                         dbc.DropdownMenuItem("Clustering",href="/dimension_reduction_clustering"),
#                         dbc.DropdownMenuItem("KNN",href="/knn"),
#                           ],
#                       ),
                    ],
          brand="Home",
          brand_href="/home",
          sticky="top",
        )
    return navbar